from ronglian_sms_sdk import SmsSDK
import smtplib
from email.mime.text import MIMEText
from email.header import Header
# 短信


def send_message(message):
    accId = '8aaf07087771cb880177bf65c6fb018a'
    accToken = '903061d4983346ebae25f6431b56545f'
    appId = '8aaf07087771cb880177bf65c8790191'
    sdk = SmsSDK(accId, accToken, appId)
    tid = '1'
    mobile = '13688193177'
    datas = (message,message)
    resp = sdk.sendMessage(tid, mobile, datas)
    print(resp)

# 邮件

def send_email(subject='训练完毕',content='训练完毕'):
    host = 'smtp.qq.com'
    username = '263657041@qq.com'
    password = 'caaflyniknglbicg'

    sender = '263657041@qq.com'
    receivers = ['peterbon@qq.com']
    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header("Peter的电脑", 'utf-8')  # 发送者
    message['To'] = Header("Peter的手机", 'utf-8')  # 接收者
    message['Subject'] = Header(subject, 'utf-8')
    try:
        smtp = smtplib.SMTP_SSL(host, 465)
        smtp.login(username, password)
        smtp.sendmail(sender, receivers, message.as_string())
        smtp.quit()
        print('邮件发送成功')
    except smtplib.SMTPException:
        print('发送失败')







