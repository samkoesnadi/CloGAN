import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import urllib.request
import time

mail_content = '''
Shalom,
it is me again... I would like to inform you that Oculus Rift S is now available on the website https://www.oculus.com/rift-s/

If you are not Samuel, you are most probably Manal. Please contact him immediately! 

Thank you best friend, you are the best!!!


Best regards, 
Kobe - Samuel's AI Assistant
'''
headers = {"accept-language": "de-DE",
"referer": "https://www.oculus.com/quest/",
"sec-fetch-dest": "document",
"sec-fetch-mode": "navigate",
"sec-fetch-site": "same-origin",
"sec-fetch-user": "?1",
"upgrade-insecure-requests": "1",
"user-agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)"}

# The mail addresses and password
sender_address = 'kobe.ai.samuel@gmail.com'
sender_pass = 'dudeismine4258'
receiver_address = ['samuelmat19@gmail.com', 'manal.wmb@gmail.com']

print("Starting Loop...")

# check Nicht auf Lager content
while True:
    # get the website content
    req = urllib.request.Request("https://www.oculus.com/rift-s/", headers=headers)
    response = urllib.request.urlopen(req)
    contents = response.read().decode("utf-8")

    # check if Nicht auf Lager doesnt exist
    if contents.find(
            '''<div class="_48ag _48aj _48ak hero__buy-btn-container"><span class="OuiFBPixelClickEventWrapper" id="u_0_1"><div class="_4r6-"><button class="_3hmq _4pg_ _4phd _4phk _4r70" disabled="1" data-testid="hero-buy-now" type="submit" id="u_0_2">Nicht auf Lager</button><p class="_4r71">Leider ist diese Funktion zurzeit in deinem Land nicht verfügbar.</p></div>''') == -1:
        print("Oculus Rift S is available!!!")

        # ... send email
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = ", ".join(receiver_address)
        message['Subject'] = 'Oculus Rift S available in the website - Kobe'  # The subject line
        # The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        # Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
        session.starttls()  # enable security
        session.login(sender_address, sender_pass)  # login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()

        time.sleep(43200)  # sleep for half day

    # wait for half hour before restarting everything
    time.sleep(1800)
