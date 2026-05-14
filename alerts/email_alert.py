"""Gmail SMTP email alerts with per-session rate limiting."""
import smtplib
import time
from email.mime.text import MIMEText

DELAI_SMS = 60  # minimum seconds between consecutive emails


class EmailAlerter:
    def __init__(self, expediteur, mot_passe, destinataire):
        self.expediteur   = expediteur
        self.mot_passe    = mot_passe
        self.destinataire = destinataire
        self._dernier_envoi = 0

    def envoyer(self, message):
        if time.time() - self._dernier_envoi < DELAI_SMS:
            return
        try:
            msg = MIMEText(message)
            msg["Subject"] = "ALERTE DriverGuard"
            msg["From"]    = f"DriverGuard <{self.expediteur}>"
            msg["To"]      = self.destinataire
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(self.expediteur, self.mot_passe)
                smtp.send_message(msg)
            self._dernier_envoi = time.time()
            print(f"Email envoyé : {message}")
        except Exception as e:
            print(f"Erreur Email : {e}")
