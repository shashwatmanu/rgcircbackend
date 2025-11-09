import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

# Try to load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Email configuration from environment variables
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM", MAIL_USERNAME)
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_FROM_NAME = os.getenv("MAIL_FROM_NAME", "RGCIRC Recon Platform")

def send_verification_email(to_email: str, username: str, verification_url: str) -> bool:
    """
    Send email verification email to user
    
    Args:
        to_email: Recipient email address
        username: User's username
        verification_url: Complete verification URL with token
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        print("[Email] ℹ️ Email not sent - MAIL_USERNAME or MAIL_PASSWORD not configured")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Verify your RGCIRC account"
        msg["From"] = f"{MAIL_FROM_NAME} <{MAIL_FROM}>"
        msg["To"] = to_email
        
        # HTML email template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: #f9fafb;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .button {{
                    display: inline-block;
                    padding: 14px 32px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    color: #6b7280;
                    font-size: 12px;
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                }}
                .highlight {{
                    background: #fef3c7;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin: 0; font-size: 28px;">🏥 RGCIRC</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Reconciliation Platform</p>
            </div>
            <div class="content">
                <h2 style="color: #1f2937; margin-top: 0;">Hello {username}! 👋</h2>
                <p>Thank you for registering with RGCIRC Reconciliation Platform.</p>
                <p>To complete your registration and access all features, please verify your email address by clicking the button below:</p>
                
                <div style="text-align: center;">
                    <a href="{verification_url}" class="button">Verify Email Address</a>
                </div>
                
                <p style="color: #6b7280; font-size: 14px; margin-top: 30px;">
                    If the button doesn't work, copy and paste this link into your browser:
                </p>
                <p style="word-break: break-all; background: white; padding: 12px; border-radius: 6px; font-size: 12px; color: #4b5563;">
                    {verification_url}
                </p>
                
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 4px;">
                    <p style="margin: 0; color: #856404;">
                        <strong>⏰ Important:</strong> This verification link will expire in <span class="highlight">24 hours</span>
                    </p>
                </div>
                
                <p style="color: #6b7280; font-size: 14px; margin-top: 30px;">
                    If you didn't create an account, you can safely ignore this email.
                </p>
            </div>
            <div class="footer">
                <p>© 2025 RGCIRC Reconciliation Platform. All rights reserved.</p>
                <p>Healthcare Financial Reconciliation Made Simple</p>
            </div>
        </body>
        </html>
        """
        
        # Plain text fallback
        text_content = f"""
        Hello {username}!
        
        Thank you for registering with RGCIRC Reconciliation Platform.
        
        To complete your registration, please verify your email address by visiting:
        {verification_url}
        
        This link will expire in 24 hours.
        
        If you didn't create an account, you can safely ignore this email.
        
        ---
        RGCIRC Reconciliation Platform
        Healthcare Financial Reconciliation Made Simple
        """
        
        # Attach both HTML and plain text versions
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        print(f"[Email] 📧 Connecting to {MAIL_SERVER}:{MAIL_PORT}...")
        
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            print(f"[Email] 🔐 Authenticating as {MAIL_USERNAME}...")
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            print(f"[Email] 📤 Sending verification email to {to_email}...")
            server.send_message(msg)
            print(f"[Email] ✅ Verification email sent successfully!")
        
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌ SMTP Authentication failed - check MAIL_USERNAME and MAIL_PASSWORD")
        return False
    except smtplib.SMTPException as e:
        print(f"[Email] ❌ SMTP error: {e}")
        return False
    except Exception as e:
        print(f"[Email] ❌ Failed to send email: {e}")
        return False


def send_password_reset_email(to_email: str, username: str, reset_url: str) -> bool:
    """
    Send password reset email to user (future feature)
    
    Args:
        to_email: Recipient email address
        username: User's username
        reset_url: Complete password reset URL with token
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        print("[Email] ℹ️ Email not sent - MAIL_USERNAME or MAIL_PASSWORD not configured")
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Reset your RGCIRC password"
        msg["From"] = f"{MAIL_FROM_NAME} <{MAIL_FROM}>"
        msg["To"] = to_email
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: #f9fafb;
                    padding: 30px;
                    border-radius: 0 0 10px 10px;
                }}
                .button {{
                    display: inline-block;
                    padding: 14px 32px;
                    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    color: #6b7280;
                    font-size: 12px;
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin: 0; font-size: 28px;">🏥 RGCIRC</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Reconciliation Platform</p>
            </div>
            <div class="content">
                <h2 style="color: #1f2937; margin-top: 0;">Password Reset Request</h2>
                <p>Hello {username},</p>
                <p>We received a request to reset your password. Click the button below to create a new password:</p>
                
                <div style="text-align: center;">
                    <a href="{reset_url}" class="button">Reset Password</a>
                </div>
                
                <p style="color: #6b7280; font-size: 14px; margin-top: 30px;">
                    This link will expire in 1 hour for security reasons.
                </p>
                
                <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0; border-radius: 4px;">
                    <p style="margin: 0; color: #991b1b;">
                        <strong>⚠️ Security Notice:</strong> If you didn't request this password reset, please ignore this email and ensure your account is secure.
                    </p>
                </div>
            </div>
            <div class="footer">
                <p>© 2025 RGCIRC Reconciliation Platform. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Password Reset Request
        
        Hello {username},
        
        We received a request to reset your password. Visit this link to create a new password:
        {reset_url}
        
        This link will expire in 1 hour.
        
        If you didn't request this, please ignore this email.
        
        ---
        RGCIRC Reconciliation Platform
        """
        
        part1 = MIMEText(text_content, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"[Email] ✅ Password reset email sent to {to_email}")
        return True
        
    except Exception as e:
        print(f"[Email] ❌ Failed to send password reset email: {e}")
        return False