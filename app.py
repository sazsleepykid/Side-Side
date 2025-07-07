import streamlit as st
import os
import sys
import imaplib
import email
import email.utils
from datetime import datetime, timedelta
from collections import defaultdict
import json
import re
from bs4 import BeautifulSoup

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from vector_store import load_vector_store, setup_vector_store, check_vector_store_exists
from llama_model import initialize_llama_model, check_gpu_availability
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import GPUEmbeddings
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


def clean_html(html_text):
    """Convert HTML to readable plain text"""
    try:
        return BeautifulSoup(html_text, "html.parser").get_text(separator="\n")
    except Exception:
        return html_text  # fallback

def fetch_emails(mail, mailbox, start_date, end_date, max_emails=100, progress_bar=None):
    """Fetch emails from the selected mailbox within date range"""
    try:
        if not refresh_mail_connection():
            return []
            
        mail.select(mailbox)
        start_date_str = start_date.strftime("%d-%b-%Y")
        end_date_str = end_date.strftime("%d-%b-%Y")
        search_criteria = f'(SINCE "{start_date_str}" BEFORE "{end_date_str}")'
        status, data = mail.search(None, search_criteria)
        if status != 'OK':
            return []
        
        email_ids = data[0].split()
        total_emails = len(email_ids)
        
        if progress_bar:
            progress_bar.text(f"Found {total_emails} emails. Processing...")

        if max_emails and total_emails > max_emails:
            email_ids = email_ids[:max_emails]
        
        emails = []
        for i, email_id in enumerate(email_ids):
            try:
                if progress_bar:
                    progress_bar.progress((i + 1) / len(email_ids))
                    progress_bar.text(f"Processing email {i+1}/{len(email_ids)}...")
                
                if i > 0 and i % 10 == 0:
                    if not refresh_mail_connection():
                        break
                
                status, data = mail.fetch(email_id, '(RFC822)')
                if status != 'OK':
                    continue
                
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)

                from_header = msg.get('From', 'Unknown <unknown@example.com>')
                if '<' in from_header and '>' in from_header:
                    sender_name = from_header.split('<')[0].strip().strip('"')
                    sender_email = from_header.split('<')[1].split('>')[0].strip()
                else:
                    sender_name = "Unknown"
                    sender_email = from_header.strip()

                subject = msg.get('Subject', 'No Subject')
                date_str = msg.get('Date')
                try:
                    date_tuple = email.utils.parsedate_tz(date_str)
                    received_time = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple)) if date_tuple else datetime.now()
                except:
                    received_time = datetime.now()

                body_text = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if "attachment" in content_disposition:
                            continue
                        try:
                            part_payload = part.get_payload(decode=True).decode(errors="ignore")
                            if content_type == "text/html":
                                body_text += clean_html(part_payload)
                            elif content_type == "text/plain":
                                body_text += part_payload
                        except:
                            continue
                else:
                    try:
                        payload = msg.get_payload(decode=True).decode(errors="ignore")
                        content_type = msg.get_content_type()
                        body_text = clean_html(payload) if content_type == "text/html" else payload
                    except:
                        body_text = msg.get_payload()

                email_obj = {
                    "id": email_id.decode(),
                    "sender_name": sender_name,
                    "sender_email": sender_email,
                    "subject": subject,
                    "date": received_time,
                    "body": body_text.strip(),
                    "raw_email": raw_email
                }
                emails.append(email_obj)

            except Exception as e:
                st.error(f"Error processing email {i+1}: {e}")
        
        return emails

    except Exception as e:
        st.error(f"Error fetching emails: {e}")
        return []

def create_vector_store_from_emails(emails, progress_bar=None):
    """Create a vector store from fetched emails in-memory using GPU-based embeddings"""
    if progress_bar:
        progress_bar.text("Preparing email documents...")
        progress_bar.progress(0.1)

    texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for i, email_obj in enumerate(emails):
        doc_text = f"Subject: {email_obj['subject']}\n"
        doc_text += f"From: {email_obj['sender_name']} <{email_obj['sender_email']}>\n"
        doc_text += f"Date: {email_obj['date'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        doc_text += email_obj['body']
        chunks = text_splitter.split_text(doc_text)
        texts.extend(chunks)

    if progress_bar:
        progress_bar.text("Creating embeddings...")
        progress_bar.progress(0.6)

    # ✅ GPU-accelerated embeddings
    embeddings = GPUEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    if progress_bar:
        progress_bar.text("Building vector store...")
        progress_bar.progress(0.8)

    vector_store = Chroma.from_texts(texts=texts, embedding=embeddings)

    if progress_bar:
        progress_bar.text("Vector store created successfully!")
        progress_bar.progress(1.0)

    return vector_store

def load_llm_model(model_path="models/llama-2-7b-chat.Q4_K_M.gguf"):
    """Load the LLM model"""
    try:
        # Check GPU availability
        gpu_available, gpu_info = check_gpu_availability()
        
        # Initialize model with GPU if available
        llm = initialize_llama_model(model_path=model_path, n_ctx=2048)
        return llm, True, None
    except Exception as e:
        return None, False, str(e)

# Login form
def show_login_form():
    st.title("📧 Email Assistant")
    
    # Check GPU status and display in sidebar
    with st.sidebar:
        st.subheader("System Info")
        gpu_available, gpu_info = check_gpu_availability()
        if gpu_available:
            st.success(f"✅ GPU Acceleration: {gpu_info}")
        else:
            st.warning(f"⚠️ {gpu_info} - Running on CPU")
    
    with st.form("login_form"):
        st.subheader("Connect to your email")
        
        # Default values for your account
        default_email = "pa.sasitheran@cre8iot.com"
        default_server = "np134.mschosting.cloud"
        default_port = 993
        
        email_address = st.text_input("Email Address", value=default_email)
        password = st.text_input("Password", type="password")
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
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
            if not email_address:
                st.error("Email address is required")
                return
                
            if not password:
                st.error("Password is required")
                return
            
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
                        progress_placeholder = st.empty()
                        progress_bar = progress_placeholder.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Import email_reader functions
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("email_reader", "email_reader.py")
                            email_reader = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(email_reader)
                            
                            # Calculate date range
                            end_date = datetime.now()
                            start_date = end_date - timedelta(days=days_back)
                            
                            status_text.text("Connecting to email server...")
                            
                            # Define progress callback
                            def update_progress(current, total, message=None):
                                progress = current / total if total > 0 else 0
                                progress_bar.progress(progress)
                                if message:
                                    status_text.text(message)
                                else:
                                    status_text.text(f"Processing email {current}/{total}...")
                            
                            # Connect to email (reuse existing connection)
                            mail_reader, _ = email_reader.connect_imap_inbox(
                                email_address=email_address,
                                password=password,
                                imap_server=imap_server,
                                imap_port=imap_port,
                                silent_mode=True
                            )
                            
                            status_text.text("Fetching emails...")
                            progress_bar.progress(0.1)
                            
                            # Fetch emails
                            email_ids = email_reader.fetch_emails_by_date(
                                mail_reader, start_date, end_date, max_emails, silent_mode=True
                            )
                            
                            if email_ids:
                                status_text.text(f"Found {len(email_ids)} emails. Processing...")
                                progress_bar.progress(0.2)
                                
                                # Group and save emails
                                grouped = email_reader.group_emails_by_sender(
                                    mail_reader, 
                                    email_ids, 
                                    silent_mode=True,
                                    progress_callback=update_progress
                                )
                                
                                status_text.text("Saving emails and attachments...")
                                progress_bar.progress(0.6)
                                
                                email_metadata = email_reader.save_emails_and_attachments(
                                    grouped, 
                                    silent_mode=True,
                                    progress_callback=update_progress
                                )
                                
                                # Close the connection
                                try:
                                    mail_reader.close()
                                    mail_reader.logout()
                                except:
                                    pass
                                
                                if email_metadata:
                                    status_text.text(f"Successfully extracted {len(email_metadata)} emails! Creating searchable database...")
                                    progress_bar.progress(0.8)
                                    
                                    # Create vector store
                                    setup_vector_store(
                                        silent_mode=True,
                                        progress_callback=lambda progress, message: (
                                            progress_bar.progress(0.8 + progress * 0.2),
                                            status_text.text(message)
                                        )
                                    )
                                    
                                    progress_bar.progress(1.0)
                                    status_text.text("Email database created successfully!")
                                    st.success(f"Successfully processed {len(email_metadata)} emails!")
                                else:
                                    status_text.text("No emails were saved.")
                                    st.warning("No emails were saved.")
                            else:
                                status_text.text("No emails found in the specified date range.")
                                st.warning("No emails found in the specified date range.")
                                
                            
                        except Exception as e:
                            status_text.text(f"Error: {str(e)}")
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
    
    # Check GPU status and display in sidebar
    with st.sidebar:
        st.subheader("System Info")
        gpu_available, gpu_info = check_gpu_availability()
        if gpu_available:
            st.success(f"✅ GPU Acceleration: {gpu_info}")
        else:
            st.warning(f"⚠️ {gpu_info} - Running on CPU")
    
    # Sidebar for mailbox selection and date range
    with st.sidebar:
        st.subheader("Email Settings")
        
        # Get mailboxes
        if "mailboxes" not in st.session_state:
            with st.spinner("Loading mailboxes..."):
                st.session_state.mailboxes = get_mailboxes(st.session_state.mail)

        # Mailbox selection - find INBOX index
        inbox_index = 0  # Default to first item
        if "mailboxes" in st.session_state and st.session_state.mailboxes:
            for i, mailbox in enumerate(st.session_state.mailboxes):
                if mailbox.upper() == "INBOX":
                    inbox_index = i
                    break

        selected_mailbox = st.selectbox(
            "Select Mailbox", 
            options=st.session_state.mailboxes,
            index=inbox_index
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
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            status_text = st.empty()
            
            with st.spinner("Fetching emails..."):
                status_text.text("Fetching emails...")
                emails = fetch_emails(
                    st.session_state.mail,
                    selected_mailbox,
                    start_date,
                    end_date + timedelta(days=1),  # Include end date
                    max_emails,
                    progress_bar=status_text
                )
                
                if emails:
                    st.session_state.emails = emails
                    
                    # Create vector store from emails
                    status_text.text("Creating searchable database...")
                    vector_store = create_vector_store_from_emails(emails, progress_bar=progress_placeholder)
                    st.session_state.vector_store = vector_store
                    
                    # Load LLM model if not already loaded
                    if st.session_state.qa_system is None:
                        status_text.text("Loading AI model...")
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
                    
                    progress_placeholder.empty()
                    status_text.empty()
                    st.success(f"Fetched {len(emails)} emails!")
                    st.rerun()
                else:
                    progress_placeholder.empty()
                    status_text.empty()
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
    if "emails" in st.session_state and st.session_state.emails:
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
        # Check if we have a vector store from previous extraction
        if check_vector_store_exists():
            # Try to load existing vector store and model
            try:
                with st.spinner("Loading existing email database..."):
                    vector_store = load_vector_store(silent_mode=True)
                    st.session_state.vector_store = vector_store
                    
                    # Load LLM model
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
                        
                        st.success("Loaded existing email database!")
                        st.session_state.emails = [{}]  # Dummy value to trigger chat interface
                        st.rerun()
                    else:
                        st.error(f"Failed to load AI model: {error}")
            except Exception as e:
                st.error(f"Error loading existing database: {e}")
                st.info("👈 Use the sidebar to select a mailbox and date range, then click 'Fetch Emails' to load your emails.")
        else:
            # Show instructions if no emails fetched yet
            st.info("👈 Use the sidebar to select a mailbox and date range, then click 'Fetch Emails' to load your emails.")

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