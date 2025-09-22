
import os
import re
import email
import imaplib
import email.mime.multipart
import email.mime.text
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import getpass

# --- Utility Functions ---
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name or "no_subject")

def connect_imap_inbox(email_address=None, password=None, imap_server=None, imap_port=993, silent_mode=False):
    """Connect to email inbox using IMAP"""
    # Require email and password
    if not email_address:
        if silent_mode:
            raise ValueError("Email address is required")
        email_address = input("Enter your email address: ")
    
    if not password:
        if silent_mode:
            raise ValueError("Password is required")
        password = getpass.getpass(f"Enter password for {email_address}: ")
    
    # Default server if not provided
    if not imap_server:
        # Try to guess IMAP server from email domain
        domain = email_address.split('@')[1]
        if domain == 'gmail.com':
            imap_server = 'imap.gmail.com'
        elif domain == 'yahoo.com':
            imap_server = 'imap.mail.yahoo.com'
        elif domain == 'outlook.com' or domain == 'hotmail.com':
            imap_server = 'outlook.office365.com'
        elif domain == 'cre8iot.com':
            imap_server = 'np134.mschosting.cloud'
        else:
            if silent_mode:
                raise ValueError(f"IMAP server for {domain} is required")
            imap_server = input(f"Enter IMAP server for {domain} (e.g., imap.{domain}): ")
    
    if not silent_mode:
        print(f"Connecting to {imap_server}:{imap_port} as {email_address}...")
    
    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL(imap_server, imap_port)
    
    try:
        mail.login(email_address, password)
        if not silent_mode:
            print(f"Successfully logged in as {email_address}")
        
        # List available mailboxes/folders (for information only)
        status, mailboxes = mail.list()
        mailbox_list = []
        
        if not silent_mode:
            print("\nAvailable mailboxes:")
            
        for i, mailbox in enumerate(mailboxes, 1):
            try:
                # Parse mailbox name
                parts = mailbox.decode().split(' "." ')
                if len(parts) > 1:
                    mailbox_name = parts[1].strip('"')
                else:
                    mailbox_name = mailbox.decode().split('"')[-2]
                
                if not silent_mode:
                    print(f"{i}. {mailbox_name}")
                mailbox_list.append(mailbox_name)
            except:
                continue
        
        # Automatically select INBOX without prompting
        selected_mailbox = "INBOX"
        
        # Find INBOX in the list (case insensitive)
        inbox_found = False
        for mailbox in mailbox_list:
            if mailbox.upper() == "INBOX":
                selected_mailbox = mailbox  # Use the exact case as found in the list
                inbox_found = True
                break
        
        mail.select(selected_mailbox)
        if not silent_mode:
            print(f"Selected mailbox: {selected_mailbox}")
        
        return mail, email_address
    
    except Exception as e:
        if not silent_mode:
            print(f"Error connecting to email: {e}")
        raise

def fetch_emails_by_date(mail, start_date, end_date, max_emails=1000, silent_mode=False):
    """Fetch emails within a date range using IMAP"""
    # Format dates for IMAP search
    start_date_str = start_date.strftime("%d-%b-%Y")
    end_date_str = end_date.strftime("%d-%b-%Y")
    
    if not silent_mode:
        print(f"Searching for emails between {start_date_str} and {end_date_str}...")
    
    # Search for emails in the date range
    search_criteria = f'(SINCE "{start_date_str}" BEFORE "{end_date_str}")'
    status, data = mail.search(None, search_criteria)
    
    if status != 'OK':
        if not silent_mode:
            print(f"Error searching for emails: {status}")
        return []
    
    # Get email IDs
    email_ids = data[0].split()
    total_emails = len(email_ids)
    
    if not silent_mode:
        print(f"Found {total_emails} emails in the date range")
    
    # Limit the number of emails to process
    if max_emails and total_emails > max_emails:
        if not silent_mode:
            print(f"Limiting to {max_emails} emails")
        email_ids = email_ids[:max_emails]
    
    return email_ids

def fetch_email_data(mail, email_id, silent_mode=False):
    """Fetch and parse a single email"""
    try:
        # Fetch email data
        status, data = mail.fetch(email_id, '(RFC822)')
        if status != 'OK':
            if not silent_mode:
                print(f"Error fetching email {email_id}: {status}")
            return None
        
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
            # Parse the date
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
        
        # Extract attachments
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition"))
                
                # Check if it's an attachment
                if "attachment" in content_disposition:
                    try:
                        filename = part.get_filename()
                        if filename:
                            attachments.append({
                                "filename": filename,
                                "content": part.get_payload(decode=True)
                            })
                    except:
                        pass
        
        # Create email object
        email_obj = {
            "id": email_id.decode() if isinstance(email_id, bytes) else email_id,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "subject": subject,
            "date": received_time,
            "date_str": date_str,
            "body": body_text,
            "attachments": attachments,
            "raw_email": raw_email,
            "msg": msg
        }
        
        return email_obj
    
    except Exception as e:
        if not silent_mode:
            print(f"Error processing email: {e}")
        return None

def group_emails_by_sender(mail, email_ids, silent_mode=False, progress_callback=None):
    """Group emails by sender"""
    grouped = defaultdict(list)
    total = len(email_ids)
    processed = 0
    errors = 0
    
    if not silent_mode:
        print(f"Processing {total} emails...")
    
    for i, email_id in enumerate(email_ids):
        try:
            if not silent_mode and i % 10 == 0:
                print(f"Processing email {i+1}/{total}...")
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, total)
            
            email_obj = fetch_email_data(mail, email_id, silent_mode)
            if email_obj:
                # Store email data
                msg_data = {
                    "email_msg": email_obj["msg"],
                    "sender_name": email_obj["sender_name"],
                    "sender_email": email_obj["sender_email"],
                    "raw_email": email_obj["raw_email"]
                }
                
                grouped[email_obj["sender_email"]].append(msg_data)
                processed += 1
            
        except Exception as e:
            errors += 1
            if not silent_mode and errors < 10:  # Limit error messages
                print(f"⚠️ Error processing email #{i + 1}: {e}")
    
    if not silent_mode:
        print(f"✅ Processing complete:")
        print(f"   - Total emails processed: {processed}")
        print(f"   - Errors: {errors}")
    
    if processed == 0:
        if not silent_mode:
            print("\n⚠️ No emails were successfully processed!")
            print("Suggestions:")
            print("1. Check your IMAP settings")
            print("2. Try a different date range")
            print("3. Try a different mailbox")
    
    return grouped

def save_emails_and_attachments(grouped_emails, output_dir="data/emails_by_sender", attachments_dir="data/attachments", docs_dir="data/email_docs", silent_mode=False, progress_callback=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(attachments_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create a metadata store for the retrieval system
    email_metadata = []
    total_saved = 0
    total_attachments = 0
    total_emails = sum(len(emails) for emails in grouped_emails.values())
    processed_so_far = 0

    for sender_email, emails in grouped_emails.items():
        # Create sender folder
        folder_name = sanitize_filename(sender_email.replace("@", "_at_"))
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        summary_lines = []
        
        # Get sender name from the first email
        sender_name = emails[0]["sender_name"]
        if not silent_mode:
            print(f"\nProcessing {len(emails)} emails from {sender_name} <{sender_email}>")
        
        for i, email_data in enumerate(emails, start=1):
            try:
                # Update progress if callback provided
                if progress_callback:
                    processed_so_far += 1
                    progress_callback(processed_so_far, total_emails)
                
                msg = email_data["email_msg"]
                subject = sanitize_filename(msg.get('Subject', 'No Subject'))
                
                # Get date from email
                date_str = msg.get('Date')
                try:
                    # Parse the date
                    date_tuple = email.utils.parsedate_tz(date_str)
                    if date_tuple:
                        received_time = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
                        formatted_date = received_time.strftime("%Y-%m-%d_%H-%M-%S")
                    else:
                        formatted_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                except:
                    formatted_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                # Save as .eml format
                email_filename = f"{formatted_date}_{i}.eml"
                email_path = os.path.join(folder_path, email_filename)
                
                with open(email_path, "wb") as f:
                    if email_data.get("raw_email"):
                        f.write(email_data["raw_email"])
                    else:
                        # If no raw email, create one
                        f.write(msg.as_bytes())
                
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
                
                # Save as .txt for easier indexing
                txt_filename = f"{formatted_date}_{i}.txt"
                txt_path = os.path.join(folder_path, txt_filename)
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Subject: {msg.get('Subject', 'No Subject')}\n")
                    f.write(f"From: {msg.get('From', 'Unknown')}\n")
                    f.write(f"To: {msg.get('To', 'Unknown')}\n")
                    f.write(f"Date: {date_str}\n\n")
                    f.write(body_text or "[No body text]")

                # Also save a copy to the docs directory for indexing
                doc_filename = f"{sender_name}_{formatted_date}_{i}.txt"
                doc_path = os.path.join(docs_dir, doc_filename)
                
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(f"Subject: {msg.get('Subject', 'No Subject')}\n")
                    f.write(f"From: {msg.get('From', 'Unknown')}\n")
                    f.write(f"To: {msg.get('To', 'Unknown')}\n")
                    f.write(f"Date: {date_str}\n\n")
                    f.write(body_text or "[No body text]")

                summary_lines.append(f"{formatted_date} - {msg.get('Subject', 'No Subject')[:60]}")
                total_saved += 1
                
                # Add to metadata for retrieval
                email_metadata.append({
                    "sender_name": sender_name,
                    "sender_email": sender_email,
                    "subject": msg.get('Subject', 'No Subject'),
                    "date": formatted_date,
                    "content": body_text,
                    "file_path": txt_path
                })

                # Save attachments if present
                if msg.is_multipart():
                    for part in msg.walk():
                        content_disposition = str(part.get("Content-Disposition"))
                        
                        # Check if it's an attachment
                        if "attachment" in content_disposition:
                            try:
                                filename = part.get_filename()
                                if filename:
                                    att_filename = sanitize_filename(filename)
                                    
                                    # Create hierarchical folder structure for attachments
                                    year_str = received_time.strftime("%Y")
                                    month_str = received_time.strftime("%m")
                                    day_str = received_time.strftime("%d")
                                    
                                    sender_att_dir = os.path.join(attachments_dir, sanitize_filename(sender_name))
                                    year_dir = os.path.join(sender_att_dir, year_str)
                                    month_dir = os.path.join(year_dir, month_str)
                                    day_dir = os.path.join(month_dir, day_str)
                                    os.makedirs(day_dir, exist_ok=True)
                                    
                                    att_path = os.path.join(day_dir, att_filename)
                                    with open(att_path, "wb") as f:
                                        f.write(part.get_payload(decode=True))
                                    
                                    summary_lines.append(f"  📎 Attachment: {att_filename} (saved to {att_path})")
                                    total_attachments += 1
                            except Exception as e:
                                if not silent_mode:
                                    print(f"⚠️ Failed to save attachment: {e}")

            except Exception as e:
                if not silent_mode:
                    print(f"⚠️ Failed to save email #{i} from {sender_email}: {e}")

        # Write summary
        summary_file = os.path.join(folder_path, "summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Sender: {sender_name} <{sender_email}>\n")
            f.write(f"Total Emails: {len(emails)}\n\n")
            f.write("Summary of Emails:\n")
            for line in summary_lines:
                f.write(f"- {line}\n")
    
    # Save metadata for retrieval system
    metadata_path = os.path.join("data", "email_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(email_metadata, f, ensure_ascii=False, indent=2)
    
    if not silent_mode:
        print(f"\n✅ Saved {total_saved} emails and {total_attachments} attachments")
    return email_metadata

from db import get_connection

def save_email_to_db(account_id, email_obj, raw_path, attachments):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO emails (account_id, email_uid, sender_name, sender_email, subject, body, date, has_attachments, raw_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        account_id,
        email_obj["id"],
        email_obj["sender_name"],
        email_obj["sender_email"],
        email_obj["subject"],
        email_obj["body"],
        email_obj["date"],
        1 if attachments else 0,
        raw_path
    ))
    email_id = cur.lastrowid

    for att in attachments:
        cur.execute("""
            INSERT INTO attachments (email_id, filename, file_path)
            VALUES (?, ?, ?)
        """, (email_id, att["filename"], att["file_path"]))

    conn.commit()
    conn.close()
    return email_id


def main():
    print("📬 Connecting to email using IMAP...")
    
    # Get email credentials
    email_address = input("Enter your email address: ")
    password = getpass.getpass(f"Enter password for {email_address}: ")
    
    # Try to guess IMAP server from email domain
    domain = email_address.split('@')[1]
    if domain == 'gmail.com':
        default_server = 'imap.gmail.com'
    elif domain == 'yahoo.com':
        default_server = 'imap.mail.yahoo.com'
    elif domain == 'outlook.com' or domain == 'hotmail.com':
        default_server = 'outlook.office365.com'
    elif domain == 'cre8iot.com':
        default_server = 'np134.mschosting.cloud'
    else:
        default_server = f'imap.{domain}'
    
    imap_server = input(f"Enter IMAP server (default: {default_server}): ") or default_server
    imap_port = int(input("Enter IMAP port (default: 993): ") or "993")
    
    try:
        mail, user_email = connect_imap_inbox(
            email_address=email_address,
            password=password,
            imap_server=imap_server,
            imap_port=imap_port
        )
    except Exception as e:
        print(f"❌ Failed to connect to email: {e}")
        print("\nPossible solutions:")
        print("1. Check your email and password")
        print("2. Make sure IMAP is enabled in your email settings")
        print("3. Verify the IMAP server and port settings")
        return

    # Define date range
    print("\nEnter date range for email extraction:")
    
    # Set default date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Default range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (last 30 days)")
    
    try:
        start_year_input = input(f"Start year (default: {start_date.year}): ")
        start_year = int(start_year_input) if start_year_input else start_date.year
        
        start_month_input = input(f"Start month (default: {start_date.month}): ")
        start_month = int(start_month_input) if start_month_input else start_date.month
        
        start_day_input = input(f"Start day (default: {start_date.day}): ")
        start_day = int(start_day_input) if start_day_input else start_date.day
        
        end_year_input = input(f"End year (default: {end_date.year}): ")
        end_year = int(end_year_input) if end_year_input else end_date.year
        
        end_month_input = input(f"End month (default: {end_date.month}): ")
        end_month = int(end_month_input) if end_month_input else end_date.month
        
        end_day_input = input(f"End day (default: {end_date.day}): ")
        end_day = int(end_day_input) if end_day_input else end_date.day
        
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day, 23, 59, 59)
    except ValueError as e:
        print(f"❌ Invalid date input: {e}")
        print("Using default date range instead.")
        # Keep the default values set earlier
    
    # Limit max emails to process
    max_emails_input = input("\nMaximum emails to process (default: 1000, enter 0 for all): ")
    max_emails = int(max_emails_input) if max_emails_input else 1000
    if max_emails == 0:
        max_emails = None  # No limit
    
    # Fetch emails
    email_ids = fetch_emails_by_date(mail, start_date, end_date, max_emails)
    
    if not email_ids:
        print("❌ No emails found in the specified date range and mailbox.")
        return
    
    # Group emails by sender
    grouped = group_emails_by_sender(mail, email_ids)
    
    if not grouped:
        print("❌ No emails were successfully processed.")
        return
    
    # Save emails and attachments
    print("💾 Saving emails and attachments...")
    email_metadata = save_emails_and_attachments(grouped)
    
    # Close the connection
    try:
        mail.close()
        mail.logout()
        print("Closed IMAP connection")
    except:
        pass
    
    if not email_metadata:
        print("❌ No emails were saved.")
        return
    
    print("\n🎉 Done! Emails have been extracted and saved.")
    print(f"Total emails processed: {len(email_metadata)}")
    print("\nNext steps:")
    print("1. Run 'python llama_model.py' to download the Llama model")
    print("2. Run 'python vector_store.py' to create the vector database")
    print("3. Run 'python app.py' to start the Streamlit interface")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()