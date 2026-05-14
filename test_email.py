import smtplib
from email.mime.text import MIMEText
from config import GMAIL_EXPEDITEUR, GMAIL_MOT_PASSE, GMAIL_DESTINATAIRE

msg = MIMEText("ALERTE DriverGuard : Tête du conducteur inclinée vers l'avant !")
msg["Subject"] = "ALERTE DriverGuard"
msg["From"]    = f"DriverGuard <{GMAIL_EXPEDITEUR}>"
msg["To"]      = GMAIL_DESTINATAIRE

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    smtp.login(GMAIL_EXPEDITEUR, GMAIL_MOT_PASSE)
    smtp.send_message(msg)

print("Email sent successfully")
