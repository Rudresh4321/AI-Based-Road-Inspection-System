# email_service.py
"""
Email Service Module for FixMyStreet AI Road Inspection System
Handles all email-related functionality including OTP generation and sending
"""

import secrets
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import streamlit as st

# Database file constant (should match core_functions.py)
DB_FILE = "road_inspection.db"


# -------------------------
# OTP Generation
# -------------------------
def generate_otp(length=6):
    """Generate a secure random OTP."""
    return "".join([str(secrets.randbelow(10)) for _ in range(length)])


# -------------------------
# Email Sending
# -------------------------
def send_otp_email(recipient_email: str, otp: str) -> bool:
    """
    Send an OTP to the user's email address using st.secrets.
    
    Args:
        recipient_email: The recipient's email address
        otp: The one-time password to send
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get sender credentials from Streamlit secrets
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]  # App Password

        # Create the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Your OTP for FixMyStreet Registration"
        message["From"] = sender_email
        message["To"] = recipient_email

        # Plain text version
        text = f"""Hi,

Your One-Time Password (OTP) for registering on FixMyStreet is: {otp}

This OTP is valid for 5 minutes.

Thank you!"""

        # HTML version
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #667eea; text-align: center;">FixMyStreet Registration</h2>
                <p>Hi,</p>
                <p>Your One-Time Password (OTP) for registering on FixMyStreet is:</p>
                <div style="text-align: center; margin: 20px 0;">
                    <span style="font-size: 32px; font-weight: bold; color: #667eea; background: #f0f0f0; padding: 10px 20px; border-radius: 5px; letter-spacing: 5px;">{otp}</span>
                </div>
                <p style="color: #e74c3c;"><strong>This OTP is valid for 5 minutes.</strong></p>
                <p>If you didn't request this OTP, please ignore this email.</p>
                <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
                <p style="font-size: 12px; color: #999; text-align: center;">
                    FixMyStreet AI Road Inspection System<br>
                    Automated Road Infrastructure Management
                </p>
            </div>
        </body>
        </html>
        """

        # Attach both versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        return True
        
    except KeyError:
        st.error("Email configuration not found in secrets. Please contact administrator.")
        return False
    except smtplib.SMTPAuthenticationError:
        st.error("Email authentication failed. Please check email credentials.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"SMTP error occurred: {e}")
        return False
    except Exception as e:
        st.error(f"Failed to send OTP email: {e}")
        return False


def send_verification_notification(recipient_email: str, upload_id: str, location: str, 
                                   road_name: str, defect_count: int, reported_date: str,
                                   admin_phone: str = None) -> bool:
    """
    Send verification notification email to the inspector who reported the defect.
    
    Args:
        recipient_email: Inspector's email address
        upload_id: The upload/report ID
        location: Location of the defect
        road_name: Name of the road
        defect_count: Number of defects detected
        reported_date: Date when the defect was reported
        admin_phone: Admin contact phone number (optional)
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get sender credentials from Streamlit secrets
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        
        # Get admin phone from secrets or use provided value
        if admin_phone is None:
            admin_phone = st.secrets.get("admin", {}).get("phone", "Not Available")

        # Create the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = f"✅ Road Defect Repaired & Verified - {upload_id}"
        message["From"] = sender_email
        message["To"] = recipient_email

        # Plain text version
        text = f"""Dear Inspector,

We are pleased to inform you that the road defect you reported has been successfully repaired and verified!

REPORT DETAILS:
-----------------
Report ID: {upload_id}
Location: {location}
Road Name: {road_name}
Defects Detected: {defect_count}
Reported Date: {reported_date}
Status: VERIFIED ✓

GRATITUDE MESSAGE:
------------------
Thank you for your diligent work in reporting this road defect. Your contribution helps us maintain safer and better roads for everyone in our community. Your vigilance and prompt reporting made it possible to address this issue efficiently.

We truly appreciate your dedication to improving our road infrastructure!

VERIFICATION & CONTACT:
-----------------------
The repair work has been completed and verified by our team. If you would like to:
- Verify the repair quality in person
- Report any concerns about the repair
- Get additional information

Please feel free to contact the admin:
Phone: 9976543210

Thank you once again for being a valued member of our road safety initiative!

Best Regards,
FixMyStreet Team
AI Road Inspection System"""

        # HTML version with enhanced styling
        html = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5; margin: 0; padding: 20px;">
            <div style="max-width: 650px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 28px;">🛣️ FixMyStreet</h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">Road Defect Status Update</p>
                </div>
                
                <!-- Success Badge -->
                <div style="background: #4caf50; color: white; padding: 20px; text-align: center;">
                    <h2 style="margin: 0; font-size: 24px;">✅ Defect Repaired & Verified!</h2>
                </div>
                
                <!-- Main Content -->
                <div style="padding: 30px;">
                    <p style="font-size: 16px; color: #555;">Dear Inspector,</p>
                    
                    <p style="font-size: 15px; color: #666; line-height: 1.8;">
                        We are <strong style="color: #4caf50;">delighted</strong> to inform you that the road defect you reported has been successfully repaired and verified by our team!
                    </p>
                    
                    <!-- Report Details Card -->
                    <div style="background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; margin: 25px 0; border-radius: 6px;">
                        <h3 style="color: #667eea; margin-top: 0; font-size: 18px;">📋 Report Details</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Report ID:</td>
                                <td style="padding: 8px 0; color: #333;"><code style="background: #e9ecef; padding: 4px 8px; border-radius: 4px; font-size: 14px;">{upload_id}</code></td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Location:</td>
                                <td style="padding: 8px 0; color: #333;">{location}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Road Name:</td>
                                <td style="padding: 8px 0; color: #333;">{road_name}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Defects Found:</td>
                                <td style="padding: 8px 0; color: #333;"><strong>{defect_count}</strong> defect(s)</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Reported Date:</td>
                                <td style="padding: 8px 0; color: #333;">{reported_date}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Status:</td>
                                <td style="padding: 8px 0;"><span style="background: #4caf50; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600;">✓ VERIFIED</span></td>
                            </tr>
                        </table>
                    </div>
                    
                    <!-- Gratitude Message -->
                    <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 20px; margin: 25px 0; border-radius: 8px; border: 2px solid #ffb74d;">
                        <h3 style="color: #e65100; margin-top: 0; font-size: 18px;">🙏 Thank You for Your Service!</h3>
                        <p style="color: #555; margin: 0; line-height: 1.8;">
                            <strong>Your contribution matters!</strong> Thank you for your diligent work in reporting this road defect. 
                            Your vigilance and prompt action help us maintain safer and better roads for everyone in our community.
                        </p>
                        <p style="color: #555; margin: 15px 0 0 0; line-height: 1.8;">
                            We truly appreciate your dedication to improving our road infrastructure. Together, we're making a real difference! 🌟
                        </p>
                    </div>
                    
                    <!-- Contact Information -->
                    <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin: 25px 0; border-radius: 6px;">
                        <h3 style="color: #1565c0; margin-top: 0; font-size: 18px;">📞 Need Further Verification?</h3>
                        <p style="color: #555; margin: 10px 0; line-height: 1.8;">
                            The repair work has been completed and verified by our team. If you would like to:
                        </p>
                        <ul style="color: #555; margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                            <li>Verify the repair quality in person</li>
                            <li>Report any concerns about the repair</li>
                            <li>Get additional information</li>
                        </ul>
                        <p style="color: #555; margin: 15px 0 0 0;">
                            <strong>Please contact the admin:</strong><br>
                            📱 Phone: <a href="tel:{admin_phone}" style="color: #2196f3; text-decoration: none; font-weight: 600;">{admin_phone}</a>
                        </p>
                    </div>
                    
                    <p style="font-size: 15px; color: #666; margin-top: 30px;">
                        Thank you once again for being a valued member of our road safety initiative!
                    </p>
                    
                    <p style="font-size: 15px; color: #666; margin-top: 20px;">
                        <strong>Best Regards,</strong><br>
                        <span style="color: #667eea; font-weight: 600;">The FixMyStreet Team</span>
                    </p>
                </div>
                
                <!-- Footer -->
                <div style="background: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6;">
                    <p style="margin: 0; color: #999; font-size: 13px;">
                        <strong>FixMyStreet AI Road Inspection System</strong><br>
                        Automated Road Infrastructure Management<br>
                        Making Roads Safer, One Report at a Time
                    </p>
                    <p style="margin: 15px 0 0 0; color: #999; font-size: 12px;">
                        This is an automated notification. Please do not reply to this email.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        # Attach both versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        return True
        
    except KeyError:
        st.error("Email configuration not found in secrets. Please contact administrator.")
        return False
    except smtplib.SMTPAuthenticationError:
        st.error("Email authentication failed. Please check email credentials.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"SMTP error occurred: {e}")
        return False
    except Exception as e:
        st.error(f"Failed to send verification notification email: {e}")
        return False


def send_notification_email(recipient_email: str, subject: str, message_body: str) -> bool:
    """
    Send a general notification email.
    
    Args:
        recipient_email: The recipient's email address
        subject: Email subject line
        message_body: Plain text message body
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get sender credentials from Streamlit secrets
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]

        # Create the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = recipient_email

        # Create HTML version with styling
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #667eea; text-align: center;">FixMyStreet Notification</h2>
                <div style="margin: 20px 0;">
                    {message_body.replace(chr(10), '<br>')}
                </div>
                <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
                <p style="font-size: 12px; color: #999; text-align: center;">
                    FixMyStreet AI Road Inspection System
                </p>
            </div>
        </body>
        </html>
        """

        # Attach both versions
        part1 = MIMEText(message_body, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        return True
        
    except Exception as e:
        st.error(f"Failed to send notification email: {e}")
        return False


# -------------------------
# OTP Database Operations
# -------------------------
def store_otp(email: str, otp: str) -> bool:
    """
    Store OTP in SQLite database.
    
    Args:
        email: User's email address
        otp: Generated OTP
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Clean up old OTPs for this email
        cursor.execute("DELETE FROM otp_tokens WHERE email = ?", (email,))
        
        # Insert new OTP
        cursor.execute(
            "INSERT INTO otp_tokens (email, otp, created_at) VALUES (?, ?, ?)",
            (email, otp, datetime.now())
        )
        
        conn.commit()
        conn.close()
        return True
        
    except sqlite3.Error as e:
        st.error(f"Database error storing OTP: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error storing OTP: {str(e)}")
        return False


def send_fixed_notification(recipient_email: str, upload_id: str, location: str, 
                           road_name: str, defect_count: int, reported_date: str,
                           admin_phone: str = None) -> bool:
    """
    Send notification email to the inspector when their reported defect is marked as Fixed.
    
    Args:
        recipient_email: Inspector's email address
        upload_id: The upload/report ID
        location: Location of the defect
        road_name: Name of the road
        defect_count: Number of defects detected
        reported_date: Date when the defect was reported
        admin_phone: Admin contact phone number (optional)
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Get sender credentials from Streamlit secrets
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        
        # Get admin phone from secrets or use provided value
        if admin_phone is None:
            admin_phone = st.secrets.get("admin", {}).get("phone", "Not Available")

        # Create the email message
        message = MIMEMultipart("alternative")
        message["Subject"] = f"✅ Road Defect Repaired - {upload_id}"
        message["From"] = sender_email
        message["To"] = recipient_email

        # Plain text version
        text = f"""Dear Inspector,

Great news! The road defect you reported has been successfully repaired!

REPORT DETAILS:
-----------------
Report ID: {upload_id}
Location: {location}
Road Name: {road_name}
Defects Detected: {defect_count}
Reported Date: {reported_date}
Status: FIXED ✓

NEXT STEPS:
-----------
The repair work has been completed. The report is now awaiting final verification by our admin team.

GRATITUDE MESSAGE:
------------------
Thank you for your diligent work in reporting this road defect. Your contribution helps us maintain safer and better roads for everyone in our community. Your vigilance and prompt reporting made it possible to address this issue efficiently.

We truly appreciate your dedication to improving our road infrastructure!

QUESTIONS OR CONCERNS?
----------------------
If you have any questions about the repair or would like to verify the work yourself, please feel free to contact the admin:

Phone: {admin_phone}

Thank you once again for being a valued member of our road safety initiative!

Best Regards,
FixMyStreet Team
AI Road Inspection System"""

        # HTML version with enhanced styling
        html = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f5f5f5; margin: 0; padding: 20px;">
            <div style="max-width: 650px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 28px;">🛣️ FixMyStreet</h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 16px;">Road Defect Status Update</p>
                </div>
                
                <!-- Success Badge -->
                <div style="background: #ff9800; color: white; padding: 20px; text-align: center;">
                    <h2 style="margin: 0; font-size: 24px;">✅ Defect Repaired!</h2>
                    <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.95;">Awaiting Final Verification</p>
                </div>
                
                <!-- Main Content -->
                <div style="padding: 30px;">
                    <p style="font-size: 16px; color: #555;">Dear Inspector,</p>
                    
                    <p style="font-size: 15px; color: #666; line-height: 1.8;">
                        <strong style="color: #ff9800;">Great news!</strong> The road defect you reported has been successfully repaired by our maintenance team!
                    </p>
                    
                    <!-- Report Details Card -->
                    <div style="background: #f8f9fa; border-left: 4px solid #667eea; padding: 20px; margin: 25px 0; border-radius: 6px;">
                        <h3 style="color: #667eea; margin-top: 0; font-size: 18px;">📋 Report Details</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Report ID:</td>
                                <td style="padding: 8px 0; color: #333;"><code style="background: #e9ecef; padding: 4px 8px; border-radius: 4px; font-size: 14px;">{upload_id}</code></td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Location:</td>
                                <td style="padding: 8px 0; color: #333;">{location}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Road Name:</td>
                                <td style="padding: 8px 0; color: #333;">{road_name}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Defects Found:</td>
                                <td style="padding: 8px 0; color: #333;"><strong>{defect_count}</strong> defect(s)</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Reported Date:</td>
                                <td style="padding: 8px 0; color: #333;">{reported_date}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #666; font-weight: 600;">Status:</td>
                                <td style="padding: 8px 0;"><span style="background: #ff9800; color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600;">✓ FIXED</span></td>
                            </tr>
                        </table>
                    </div>
                    
                    <!-- Next Steps -->
                    <div style="background: #fff3e0; border-left: 4px solid #ff9800; padding: 20px; margin: 25px 0; border-radius: 6px;">
                        <h3 style="color: #e65100; margin-top: 0; font-size: 18px;">📌 Next Steps</h3>
                        <p style="color: #555; margin: 0; line-height: 1.8;">
                            The repair work has been <strong>completed</strong>. The report is now awaiting <strong>final verification</strong> by our admin team to ensure the repair meets quality standards.
                        </p>
                    </div>
                    
                    <!-- Gratitude Message -->
                    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 20px; margin: 25px 0; border-radius: 8px; border: 2px solid #66bb6a;">
                        <h3 style="color: #2e7d32; margin-top: 0; font-size: 18px;">🙏 Thank You for Your Service!</h3>
                        <p style="color: #555; margin: 0; line-height: 1.8;">
                            <strong>Your contribution matters!</strong> Thank you for your diligent work in reporting this road defect. 
                            Your vigilance and prompt action help us maintain safer and better roads for everyone in our community.
                        </p>
                        <p style="color: #555; margin: 15px 0 0 0; line-height: 1.8;">
                            We truly appreciate your dedication to improving our road infrastructure. Together, we're making a real difference! 🌟
                        </p>
                    </div>
                    
                    <!-- Contact Information -->
                    <div style="background: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin: 25px 0; border-radius: 6px;">
                        <h3 style="color: #1565c0; margin-top: 0; font-size: 18px;">❓ Questions or Concerns?</h3>
                        <p style="color: #555; margin: 10px 0; line-height: 1.8;">
                            If you have any questions about the repair or would like to verify the work yourself, please feel free to contact the admin:
                        </p>
                        <p style="color: #555; margin: 15px 0 0 0;">
                            📱 <strong>Admin Phone:</strong> <a href="tel:{admin_phone}" style="color: #2196f3; text-decoration: none; font-weight: 600;">{admin_phone}</a>
                        </p>
                    </div>
                    
                    <p style="font-size: 15px; color: #666; margin-top: 30px;">
                        Thank you once again for being a valued member of our road safety initiative!
                    </p>
                    
                    <p style="font-size: 15px; color: #666; margin-top: 20px;">
                        <strong>Best Regards,</strong><br>
                        <span style="color: #667eea; font-weight: 600;">The FixMyStreet Team</span>
                    </p>
                </div>
                
                <!-- Footer -->
                <div style="background: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6;">
                    <p style="margin: 0; color: #999; font-size: 13px;">
                        <strong>FixMyStreet AI Road Inspection System</strong><br>
                        Automated Road Infrastructure Management<br>
                        Making Roads Safer, One Report at a Time
                    </p>
                    <p style="margin: 15px 0 0 0; color: #999; font-size: 12px;">
                        This is an automated notification. Please do not reply to this email.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        # Attach both versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        
        return True
        
    except KeyError:
        st.error("Email configuration not found in secrets. Please contact administrator.")
        return False
    except smtplib.SMTPAuthenticationError:
        st.error("Email authentication failed. Please check email credentials.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"SMTP error occurred: {e}")
        return False
    except Exception as e:
        st.error(f"Failed to send fixed notification email: {e}")
        return False
    
def verify_otp(email: str, otp: str) -> bool:
    """
    Verify OTP from SQLite database.
    
    Args:
        email: User's email address
        otp: OTP to verify
        
    Returns:
        bool: True if OTP is valid, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if OTP exists, is unused, and not expired (5 minutes)
        cursor.execute("""
            SELECT id FROM otp_tokens 
            WHERE email = ? AND otp = ? AND is_used = FALSE 
            AND datetime(created_at, '+5 minutes') > datetime('now')
        """, (email, otp))
        
        result = cursor.fetchone()
        
        if result:
            # Mark OTP as used
            cursor.execute(
                "UPDATE otp_tokens SET is_used = TRUE WHERE id = ?",
                (result[0],)
            )
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False
        
    except sqlite3.Error as e:
        st.error(f"Database error verifying OTP: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error verifying OTP: {str(e)}")
        return False


def cleanup_expired_otps() -> int:
    """
    Clean up expired OTPs from database.
    
    Returns:
        int: Number of OTPs deleted
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Delete OTPs older than 5 minutes
        cursor.execute("""
            DELETE FROM otp_tokens 
            WHERE datetime(created_at, '+5 minutes') < datetime('now')
        """)
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
        
    except Exception as e:
        st.error(f"Error cleaning up expired OTPs: {str(e)}")
        return 0


# -------------------------
# Email Validation
# -------------------------
def is_valid_email(email: str) -> bool:
    """
    Basic email validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email format is valid, False otherwise
    """
    import re
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_gmail_address(email: str) -> bool:
    """
    Check if email is a Gmail address.
    
    Args:
        email: Email address to check
        
    Returns:
        bool: True if Gmail address, False otherwise
    """
    return email.lower().endswith('@gmail.com')


# -------------------------
# Test Functions
# -------------------------
def test_email_configuration() -> bool:
    """
    Test email configuration without sending actual email.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        
        # Test connection
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
        
        return True
        
    except KeyError:
        st.error("Email secrets not configured properly.")
        return False
    except smtplib.SMTPAuthenticationError:
        st.error("Email authentication failed. Check credentials.")
        return False
    except Exception as e:
        st.error(f"Email configuration test failed: {e}")
        return False


if __name__ == "__main__":
    # Test functionality when run directly
    print("Email Service Module - Testing")
    print("=" * 50)
    
    # Test OTP generation
    test_otp = generate_otp()
    print(f"Generated OTP: {test_otp}")
    print(f"OTP Length: {len(test_otp)}")
    
    # Test email validation
    test_emails = [
        "user@gmail.com",
        "invalid-email",
        "test@domain.co.in",
        "bad@email"
    ]
    
    print("\nEmail Validation Tests:")
    for email in test_emails:
        valid = is_valid_email(email)
        is_gmail = is_gmail_address(email)
        print(f"  {email}: Valid={valid}, Gmail={is_gmail}")