import streamlit as st
import os
import sys
import imaplib
import email
import email.utils
from datetime import datetime, timedelta
from collections import defaultdict
import getpass
import json
import re

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from vector_store import load_vector_store, setup_vector_store
from llama_model import initialize_llama_model
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

# Set page configuration
st.set_page_config(
    page_title="Email Assistant",
    page_icon="📧",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "emails" not in st.session_state:
    st.session_state.emails = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

# Utility functions
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name or "no_subject")

def connect_to_email(email_address, password, imap_server, imap_port=993):
    """Connect to email server using IMAP"""
    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(email_address, password)
        return mail, True, None
    except Exception as e:
        return None, False, str(e)

def refresh_mail_connection():
    """Refresh the mail connection if needed"""
    try:
        # Try a simple command to check if connection is still alive
        st.session_state.mail.noop()
    except Exception as e:
        # Connection is dead, reconnect
        st.info("Reconnecting to email server...")
        mail, success, error = connect_to_email(
            st.session_state.email_address,
            st.session_state.password,
            st.session_state.imap_server,
            st.session_state.imap_port
        )
        
        if success:
            st.session_state.mail = mail
            return True
        else:
            st.error(f"Failed to reconnect: {error}")
            return False
    return True

def get_mailboxes(mail):
    """Get list of mailboxes/folders"""
    try:
        status, mailboxes = mail.list()
        mailbox_list = []
        
        for mailbox in mailboxes:
            try:
                # Parse mailbox name
                parts = mailbox.decode().split(' "." ')
                if len(parts) > 1:
                    mailbox_name = parts[1].strip('"')
                else:
                    mailbox_name = mailbox.decode().split('"')[-2]
                
                mailbox_list.append(mailbox_name)
            except:
                continue
        
        return mailbox_list
    except Exception as e:
        st.error(f"Error getting mailboxes: {e}")
        return ["INBOX"]  # Return at least INBOX as a fallback

def fetch_emails(mail, mailbox, start_date, end_date, max_emails=100):
    """Fetch emails from the selected mailbox within date range"""
    try:
        # Refresh connection if needed
        if not refresh_mail_connection():
            return []
            
        mail.select(mailbox)
        
        # Format dates for IMAP search
        start_date_str = start_date.strftime("%d-%b-%Y")
        end_date_str = end_date.strftime("%d-%b-%Y")
        
        # Search for emails in the date range
        search_criteria = f'(SINCE "{start_date_str}" BEFORE "{end_date_str}")'
        status, data = mail.search(None, search_criteria)
        
        if status != 'OK':
            return []
        
        # Get email IDs
        email_ids = data[0].split()
        total_emails = len(email_ids)
        
        # Limit the number of emails to process
        if max_emails and total_emails > max_emails:
            email_ids = email_ids[:max_emails]
        
        emails = []
        for i, email_id in enumerate(email_ids):
            try:
                # Refresh connection every 10 emails to prevent timeouts
                if i > 0 and i % 10 == 0:
                    if not refresh_mail_connection():
                        break
                
                # Fetch email data
                status, data = mail.fetch(email_id, '(RFC822)')
                if status != 'OK':
                    continue
                
                # Parse email
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                # Get sender information
                from_header = msg.get('From', 'Unknown <unknown@example.com>')
                
                # Extract email and name
                if '<' in from_header and '>' in from_header:
                    sender_name = from_header.split('<')[0].strip().strip('"')
                    sender_email = from_header.split('<')[1].split('>')[0].strip()
                else:
                    sender_name = "Unknown"
                    sender_email = from_header.strip()
                
                # Get subject
                subject = msg.get('Subject', 'No Subject')
                
                # Get date
                date_str = msg.get('Date')
                try:
                    date_tuple = email.utils.parsedate_tz(date_str)
                    if date_tuple:
                        received_time = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                    else:
                        received_time = datetime.now()
                except:
                    received_time = datetime.now()
                
                # Extract body text
                body_text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        
                        # Skip attachments
                        if "attachment" in content_disposition:
                            continue
                        
                        # Get text content
                        if content_type == "text/plain" or content_type == "text/html":
                            try:
                                body_text += part.get_payload(decode=True).decode()
                            except:
                                pass
                else:
                    try:
                        body_text = msg.get_payload(decode=True).decode()
                    except:
                        body_text = msg.get_payload()
                
                # Create email object
                email_obj = {
                    "id": email_id.decode(),
                    "sender_name": sender_name,
                    "sender_email": sender_email,
                    "subject": subject,
                    "date": received_time,
                    "body": body_text,
                    "raw_email": raw_email
                }
                
                emails.append(email_obj)
                
            except Exception as e:
                st.error(f"Error processing email: {e}")
        
        return emails
    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def create_vector_store_from_emails(emails):
    """Create a vector store from fetched emails"""
    # Create temporary documents
    temp_dir = "temp_emails"
    os.makedirs(temp_dir, exist_ok=True)
    
    documents = []
    for i, email_obj in enumerate(emails):
        # Create document text
        doc_text = f"Subject: {email_obj['subject']}\n"
        doc_text += f"From: {email_obj['sender_name']} <{email_obj['sender_email']}>\n"
        doc_text += f"Date: {email_obj['date'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        doc_text += email_obj['body']
        
        # Save to temporary file
        file_path = os.path.join(temp_dir, f"email_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc_text)
        
        # Create document
        documents.append({"path": file_path, "content": doc_text})
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            texts.append(chunk)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create in-memory vector store
    vector_store = Chroma.from_texts(texts=texts, embedding=embeddings)
    
    return vector_store

@st.cache_resource
def load_llm_model(model_path="models/llama-2-7b-chat.Q4_K_M.gguf"):
    """Load the LLM model"""
    try:
        llm = initialize_llama_model(model_path=model_path, n_ctx=2048)
        return llm, True, None
    except Exception as e:
        return None, False, str(e)

# Login form
def show_login_form():
    st.title("📧 Email Assistant")
    
    with st.form("login_form"):
        st.subheader("Connect to your email")
        
        # Default values for your account
        default_email = "pa.sasitheran@cre8iot.com"
        default_server = "np134.mschosting.cloud"
        default_port = 993
        
        email_address = st.text_input("Email Address", value=default_email)
        password = st.text_input("Password", type="password")
        imap_server = st.text_input("IMAP Server", value=default_server)
        imap_port = st.number_input("IMAP Port", value=default_port)
        
        # Add options for email extraction
        extract_emails = st.checkbox("Extract emails after login", value=True)
        
        if extract_emails:
            col1, col2 = st.columns(2)
            with col1:
                days_back = st.number_input("Days to look back", min_value=1, max_value=365, value=30)
            with col2:
                max_emails = st.number_input("Max emails to extract", min_value=10, max_value=5000, value=1000)
        
        submitted = st.form_submit_button("Connect")
        
        if submitted:
            with st.spinner("Connecting to email server..."):
                mail, success, error = connect_to_email(
                    email_address, password, imap_server, imap_port
                )
                
                if success:
                    st.session_state.mail = mail
                    st.session_state.email_address = email_address
                    st.session_state.password = password
                    st.session_state.imap_server = imap_server
                    st.session_state.imap_port = imap_port
                    st.session_state.logged_in = True
                    st.success("Successfully connected to email server!")
                    
                    # Run email extraction if selected
                    if extract_emails:
                        with st.spinner("Extracting emails... This may take a while."):
                            try:
                                # Import email_reader functions
                                import importlib.util
                                spec = importlib.util.spec_from_file_location("email_reader", "email_reader.py")
                                email_reader = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(email_reader)
                                
                                # Calculate date range
                                end_date = datetime.now()
                                start_date = end_date - timedelta(days=days_back)
                                
                                # Connect to email (reuse existing connection)
                                mail_reader, _ = email_reader.connect_imap_inbox(
                                    email_address=email_address,
                                    password=password,
                                    imap_server=imap_server,
                                    imap_port=imap_port
                                )
                                
                                # Fetch emails
                                email_ids = email_reader.fetch_emails_by_date(
                                    mail_reader, start_date, end_date, max_emails
                                )
                                
                                if email_ids:
                                    # Group and save emails
                                    grouped = email_reader.group_emails_by_sender(mail_reader, email_ids)
                                    email_metadata = email_reader.save_emails_and_attachments(grouped)
                                    
                                    # Close the connection
                                    try:
                                        mail_reader.close()
                                        mail_reader.logout()
                                    except:
                                        pass
                                    
                                    if email_metadata:
                                        st.success(f"Successfully extracted {len(email_metadata)} emails!")
                                        
                                        # Create vector store
                                        with st.spinner("Creating searchable database..."):
                                            # Import vector_store functions
                                            spec = importlib.util.spec_from_file_location("vector_store", "vector_store.py")
                                            vector_store_module = importlib.util.module_from_spec(spec)
                                            spec.loader.exec_module(vector_store_module)
                                            
                                            # Setup vector store
                                            vector_store_module.setup_vector_store()
                                            st.success("Email database created successfully!")
                                    else:
                                        st.warning("No emails were saved.")
                                else:
                                    st.warning("No emails found in the specified date range.")
                            except Exception as e:
                                st.error(f"Error extracting emails: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                    
                    st.rerun()
                else:
                    st.error(f"Failed to connect: {error}")

# Email browsing and search interface
def show_email_interface():
    st.title("📧 Email Assistant")
    st.write(f"Connected as: {st.session_state.email_address}")
    
    # Check if connection is still alive
    if not refresh_mail_connection():
        st.error("Lost connection to email server. Please log in again.")
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Sidebar for mailbox selection and date range
    with st.sidebar:
        st.subheader("Email Settings")
        
        # Get mailboxes
        if "mailboxes" not in st.session_state:
            with st.spinner("Loading mailboxes..."):
                st.session_state.mailboxes = get_mailboxes(st.session_state.mail)

        # Mailbox selection
        selected_mailbox = st.selectbox(
            "Select Mailbox", 
            options=st.session_state.mailboxes,
            index=0 if "INBOX" in st.session_state.mailboxes else 0
        )
        
        # Date range
        st.subheader("Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=start_date)
        with col2:
            end_date = st.date_input("End Date", value=end_date)
        
        # Max emails
        max_emails = st.slider("Max Emails to Fetch", min_value=10, max_value=500, value=100, step=10)
        
        # Fetch button
        if st.button("Fetch Emails"):
            with st.spinner("Fetching emails..."):
                emails = fetch_emails(
                    st.session_state.mail,
                    selected_mailbox,
                    start_date,
                    end_date + timedelta(days=1),  # Include end date
                    max_emails
                )
                
                if emails:
                    st.session_state.emails = emails
                    
                    # Create vector store from emails
                    with st.spinner("Creating searchable database..."):
                        vector_store = create_vector_store_from_emails(emails)
                        st.session_state.vector_store = vector_store
                        
                        # Load LLM model if not already loaded
                        if st.session_state.qa_system is None:
                            with st.spinner("Loading AI model..."):
                                llm, success, error = load_llm_model()
                                
                                if success:
                                    # Create QA system
                                    qa_system = RetrievalQA.from_chain_type(
                                        llm=llm,
                                        chain_type="stuff",
                                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                                        return_source_documents=True
                                    )
                                    st.session_state.qa_system = qa_system
                                else:
                                    st.error(f"Failed to load AI model: {error}")
                    
                    st.success(f"Fetched {len(emails)} emails!")
                    st.rerun()
                else:
                    st.warning("No emails found in the selected date range.")
        
        # Logout button
        if st.button("Logout"):
            # Close connection
            try:
                st.session_state.mail.close()
                st.session_state.mail.logout()
            except:
                pass
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.rerun()
    
    # Main area - show emails or chat interface
    if st.session_state.emails:
        # Show chat interface if we have emails and QA system
        if st.session_state.qa_system:
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("View source emails"):
                            st.markdown(message["sources"])
            
            # Chat input
            user_query = st.chat_input("Ask about your emails...")
            if user_query:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_query})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Searching through your emails..."):
                        response_container = st.empty()
                        result = st.session_state.qa_system({"query": user_query})
                        
                        # Format source documents
                        sources_text = ""
                        for i, doc in enumerate(result["source_documents"][:3], 1):
                            sources_text += f"**Source {i}:**\n```\n{doc.page_content[:300]}...\n```\n\n"
                        
                        # Display response
                        response_container.markdown(result["result"])
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result["result"],
                            "sources": sources_text
                        })
        else:
            st.info("AI model is loading or unavailable. Please wait or check for errors in the sidebar.")
    else:
        # Show instructions if no emails fetched yet
        st.info("👈 Use the sidebar to select a mailbox and date range, then click 'Fetch Emails' to load your emails.")
        
        # Show email stats
        if st.session_state.emails:
            st.subheader("Email Statistics")
            st.write(f"Total emails: {len(st.session_state.emails)}")
            
            # Count emails by sender
            senders = {}
            for email in st.session_state.emails:
                sender = email["sender_email"]
                if sender in senders:
                    senders[sender] += 1
                else:
                    senders[sender] = 1
            
            # Display top senders
            st.write("Top senders:")
            for sender, count in sorted(senders.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"- {sender}: {count} emails")

# Main app logic
def main():
    # Check if model exists
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
    model_exists = os.path.exists(model_path)
    
    if not model_exists:
        st.error("⚠️ Llama model not found!")
        st.warning(f"Please download the model file and place it at: {model_path}")
        st.info("You can download it from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main")
        st.stop()
    
    # Show login form or email interface based on login state
    if not st.session_state.logged_in:
        show_login_form()
    else:
        show_email_interface()

if __name__ == "__main__":
    main()