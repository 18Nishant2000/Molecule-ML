import requests
import socket
from tld import get_tld
from googlesearch import search
import re
import whois
from bs4 import BeautifulSoup
import urllib.request
import dns.resolver
from datetime import date
import joblib
import numpy as np

model1 = joblib.load('./PhishingModel/SVM.joblib')
model2 = joblib.load('./PhishingModel/RandomForest.joblib')
model3 = joblib.load('./PhishingModel/LogisticRegression.joblib')

data = []

def check(data):
    
    data = np.array(data).reshape(1, -1)
    r1 = model1.predict(data)
    r2 = model2.predict(data)
    r3 = model3.predict(data)
    
    if r1[0] + r2[0] + r3[0] >= 2:
        return 1
    else:
        return -1

def length_url(url):
    if len(url) >= 54:
        return -1
    elif 30 < len(url) < 53:
        return 0
    else:
        return 1


def long_url(url):
    result = length_url(url)
    data.append(result)


def shorten_url(url):
    try:
        if url[:4] != 'http':
            url = 'https://' + url
        response = requests.get(url,timeout=3)
        original_url = response.url
        result = length_url(original_url)
        data.append(result)
    except:
        data.append(-1)


def at_the_rate_symbol(url):
    result = 1
    for i in url:
        if i == '@':
            result = -1
            break
    if result:
        data.append(result)
    elif result == 1:
        data.append(0)


def double_slash_redirecting(url):
    try:
        index = url.index('/')
        count = url.count('/')
        if count == 2 and 5 <= index <= 6:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def prefix_suffix(url):
    result = 0
    for i in url:
        if i == '-':
            result = -1
            break
    if result:
        data.append(-1)
    else:
        data.append(1)


def port(url):
    try:
        if url[:4] != 'http':
            url = 'https://' + url
        info = get_tld(url, as_object=True)
        domain_name = info.domain
        port_no = domain_name.split(':')[1]
        if port_no:
            data.append(-1)
        else:
            data.append(1)
    except:
        data.append(-1)


def https_token(url):
    try:
        index = url.index('https')
        if index == 0:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def submitting_to_email(url):
    try:
        response = requests.get(url,timeout=3)
        mail_count = response.text.count('mail')
        mailto_count = response.text.count('mailto')
        if mail_count and mailto_count:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def links_pointing_to_page(url):
    try:
        response = requests.get(url,timeout=3).text
        number_of_links = response.count('<a href=')
        if number_of_links == 0:
            data.append(-1)
        elif number_of_links < 2:
            data.append(0)
        else:
            data.append(1)
    except:
        data.append(-1)


def google_index(url):
    try:
        site = search(url, 5)
        if site:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def popup_window(url):
    try:
        response = requests.get(url,timeout=3).text
        yes = response.count('alert(')
        if yes:
            data.append(-1)
        else:
            data.append(1)
    except:
        data.append(-1)


def right_click(url):
    try:
        response = requests.get(url,timeout=3).text
        yes = response.count('event.button==2')
        if yes:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def on_mouse_over(url):
    try:
        response = requests.get(url,timeout=3).text
        if re.findall('<script>.+onmouseover="window[.]status.+</script>', response):
            data.append(-1)
        else:
            data.append(1)
    except:
        data.append(-1)


def redirect(url):
    try:
        length_response = len(requests.get(url,timeout=3).history)
        if length_response <= 1:
            data.append(1)
        elif 2 <= length_response <= 3:
            data.append(0)
        else:
            data.append(-1)
    except:
        data.append(-1)


def abnormal_url(url):
    try:
        response = requests.get(url,timeout=3)
        if response:
            if response.text:
                data.append(1)
            else:
                data.append(-1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def having_sub_domain(url):
    try:
        if url[:4] != 'http':
            url = 'https://' + url
        info = get_tld(url, as_object=True)
        if 3 <= len(info.subdomain) <= 4:
            data.append(1)
        elif 1 <= len(info.subdomain) <= 2:
            data.append(0)
        else:
            data.append(-1)
    except:
        data.append(-1)


def having_ip_address(url):
    result = 0
    domain = url.split('.')
    for i in domain:
        if i.isdigit() or i.find('A') or i.find('B') or i.find('C') or i.find('D') or i.find('E') or i.find('F'):
            result = -1
            break
    if result:
        data.append(-1)
    else:
        data.append(1)


def web_traffic(url):
    try:
        rank = \
            BeautifulSoup(urllib.request.urlopen('http://data.alexa.com/data?cli=10&dat=s&url=' + url).read(),
                          'xml').find(
                'REACH')['RANK']
        rank = int(rank)
        if rank < 100000:
            data.append(1)
        else:
            data.append(0)
    except TypeError:
        data.append(-1)


def dns_record(url):
    try:
        domain = whois.whois(url).domain
        exists = dns.resolver.query(domain)
        if exists:
            data.append(1)
        else:
            data.append(-1)
        w = whois.whois(url)
        print(w.domain)
    except:
        data.append(-1)


def domain_registration_length(url):
    try:
        info = whois.whois(url)
        creation_date = info['creation_date']
        expiration_date = info['expiration_date']
        creation_date = [str(i) for i in creation_date]
        expiration_date = [str(i) for i in expiration_date]
        creation_date = [i[:10] for i in creation_date]
        expiration_date = [i[:10] for i in expiration_date]
        result = []
        for i in range(len(expiration_date)):
            c = int(creation_date[i].split('-')[0])
            e = int(expiration_date[i].split('-')[0])
            result.append(e - c)
        print(result)
        if result[0] > 1 and result[1] > 1:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def request_url(url):
    try:
        text = requests.get(url,timeout=3).text
        result = re.findall('"https://.*"', text)
        links = []
        for i in result:
            a = len(i)
            for j in range(a):
                link = ''
                if i[j:j + 4] == 'http':
                    link += i[j]
                    j += 1
                    while i[j] != '"':
                        link += i[j]
                        j += 1
                    links.append(link)
        c = 0
        for i in range(len(links)):
            try:
                domain = whois.whois(links[i]).domain
                url_domain = whois.whois(url).domain
                print(domain, url_domain)
                if domain == url_domain:
                    c += 1
            except:
                continue
        if len(links):
            if c > int(len(links) * 0.5):
                data.append(1)
            else:
                data.append(-1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def url_of_anchor(url):
    try:
        text = requests.get(url,timeout=3).text
        result = re.findall('<a href=.*"', text)
        links = []
        for i in result:
            a = len(i)
            for j in range(a):
                link = ''
                if i[j:j + 9] == '<a href="':
                    j += 9
                    while i[j] != '"':
                        link += i[j]
                        j += 1
                    links.append(link)
        c = 0
        for i in range(len(links)):
            try:
                domain = whois.whois(links[i]).domain
                url_domain = whois.whois(url).domain
                print(domain, url_domain)
                if domain == url_domain:
                    c += 1
            except:
                continue
        if len(links):
            if c > int(len(links) * 0.5):
                data.append(1)
            elif 0 < c < int(len(links) * 0.5):
                data.append(0)
            else:
                data.append(0)
        else:
            data.append(0)
    except:
        data.append(-1)


def link_in_tag(url):
    try:
        text = requests.get(url,timeout=3).text
        info = re.findall('<link .* href=.*"', text)
        result = []
        for i in info:
            s = ''
            for j in range(len(i)):
                if i[j:j + 6] == 'href="':
                    j += 6
                    while i[j] != '"':
                        s += i[j]
                        j += 1
                    result.append(s)
        print(result)
        c = 0
        for i in range(len(result)):
            try:
                domain = whois.whois(result[i]).domain
                url_domain = whois.whois(url).domain
                print(domain, url_domain)
                if domain == url_domain:
                    c += 1
            except:
                continue
        if len(result):
            if c > int(len(result) * 0.5):
                data.append(1)
            elif 0 < c < int(len(result) * 0.5):
                data.append(0)
            else:
                data.append(-1)
        else:
            data.append(0)
    except:
        data.append(-1)


def sfh(url):
    try:
        text = requests.get(url,timeout=3).text
        result = re.findall('<form.* action=".*"', text)
        action_address = []
        for i in result:
            s = ''
            for j in range(len(i)):
                if i[j:j + 8] == 'action="':
                    j += 8
                    while i[j] != '"':
                        s += i[j]
                        j += 1
                    action_address.append(s)
        if len(action_address):
            for i in action_address:
                if i == '' or i == ' ' or i == 'blank' or i == 'about:blank':
                    data.append(-1)
                else:
                    data.append(1)
                break
        else:
            data.append(0)
    except:
        data.append(-1)


def iframe(url):
    try:
        text = requests.get(url,timeout=3).text
        iframe_found = text.find('<iframe>') or text.find('<iframes>')
        if iframe_found:
            frameBorder_found = text.find('<frameBorder>')
            if frameBorder_found:
                data.append(1)
            else:
                data.append(-1)
        else:
            data.append(1)
    except:
        data.append(-1)


def age_of_domain(url):
    try:
        info = whois.whois(url)
        creation_date = info['creation_date']
        today = date.today()
        op = 0
        if isinstance(creation_date, type(today)):
            op = 1
            pro = str(creation_date)
            creation_date = []
            s = ''
            for i in range(10):
                s += pro[i]
            creation_date.append(s)
        elif not isinstance(creation_date, type(today)):
            op = 2
            creation_date = [str(i) for i in creation_date]
            creation_date = [i[:10] for i in creation_date]
        result = []
        today = str(today)
        t_temp = today.split('-')
        c_temp = []
        for i in range(len(creation_date)):
            c_temp.append(creation_date[i].split('-'))
        for i in range(len(c_temp)):
            cy = int(c_temp[i][0])
            cm = int(c_temp[i][1])
            ty = int(t_temp[0])
            tm = int(t_temp[1])
            result.append([ty - cy, abs(tm - cm)])
        yes = 0
        for i in result:
            if i[1] > 6:
                yes += 1
            elif i[0] > 0:
                yes += 1
        if yes:
            data.append(1)
        else:
            data.append(-1)
    except:
        data.append(-1)


def page_rank(url):
    try:
        rank = \
            BeautifulSoup(urllib.request.urlopen('http://data.alexa.com/data?cli=10&dat=s&url=' + url).read(),
                          'xml').find(
                'REACH')['RANK']
        rank = int(rank)
        if rank < 100000:
            data.append(1)
        else:
            data.append(-1)
    except TypeError:
        data.append(-1)


def ssl(url):
    try:
        response = requests.get(url,timeout=3)
        if response != '':
            data.append(1)
    except:
        data.append(-1)


def favicon(url):
    try:
        original_domain = whois.whois(url).domain
        info = requests.get(url,timeout=3).text
        r = re.findall('href=".*ico"', info)
        links = []
        for i in r:
            for j in range(len(i)):
                if i[j:j + 6] == 'href="':
                    j += 6
                    s = ''
                    while i[j] != '"':
                        s += i[j]
                        j += 1
                    links.append(s)
        y = 0
        for i in links:
            try:
                if original_domain == whois.whois(i).domain:
                    y += 1
            except:
                continue
        if len(r):
            if y > 0:
                data.append(1)
            else:
                data.append(-1)
        else:
            data.append(1)
    except:
        data.append(-1)


def statistical_report(url):
    top_ips = ['64.70.19.203', '216.218.185.162', '172.217.1.225', '175.126.123.219', '156.251.148.212',
               '54.83.43.69', '47.91.170.222', '173.230.141.80', '103.44.28.181', '103.44.28.169', '108.61.203.22',
               '23.20.239.12', '153.92.0.100', '141.8.224.221', '184.168.131.241', '122.10.109.175', '209.202.252.66',
               '199.59.242.153', '69.172.201.153', '91.227.52.108', '35.186.238.101', '185.164.136.124', '69.16.230.42',
               '18.216.20.136', '211.231.99.250', '160.121.242.52', '91.195.240.126', '156.224.121.118',
               '37.157.192.102',
               '67.227.226.240', '103.224.212.222', '52.58.78.16', '3.234.181.234', '198.11.172.242', '172.120.69.45',
               '204.95.99.26', '193.109.247.10', '52.69.166.231', '23.89.1.166', '18.211.9.206', '72.52.178.23',
               '204.11.56.48',
               '47.245.9.22', '193.109.247.224', '47.75.126.218', '156.234.215.125', '23.253.126.58', '23.236.62.147',
               '104.239.157.210', '209.99.40.223']
    top_urls = ['https://thriveglobal.com/?p=1322090&preview_id=1322090&preview...',
                'SPAMs://www.teamapp.com/clubs/537660/events/11229190...',
                'https://lifikurier.com/oplata52/banks', 'https://lifikurier.com/oplata52/ipko',
                'https://lifikurier.com/oplata52/Inteligo',
                'http://logicaingenieria.cl/cgi-bin', 'http://cliccaecontinua.com/',
                'SPAMs://thriveglobal.com/?p=1322302preview_id=1322302&preview_nonc...',
                'https://accesoriosdebebes.com/wp/',
                'SPAMs://thriveglobal.com/?p=1321770?preview_id=1321770&preview_non...',
                'https://www.adambarker.org/rsnb/cmrc.html',
                'https://online.eservice-bankaustria.at/wps/covid/userlogin...',
                'SPAMs://thriveglobal.com/?p=132decades-old&preview_id=132decades-o...',
                'SPAMs://thriveglobal.com/?p=13decades-old2&preview_id=13decades-ol...',
                'SPAMs://thriveglobal.com/?p=1322125&preview_id=1322125&preview...',
                'SPAMs://thriveglobal.com/?p=13many-years-old7&preview_id=13many-ye...',
                'SPAMs://thriveglobal.com/?p=1322248&preview_id=1322248&preview...',
                'SPAMs://thriveglobal.com/?p=1321806preview_id=1321806&preview_nonc...',
                'https://lifikurier.com/oplata52/', 'https://lifikurier.com/oplata52']
    try:
        domain = whois.whois(url).domain
        ip = 0
        ul = 0
        domain_ip = socket.gethostbyname(domain)
        for i in top_ips:
            try:
                if i == domain_ip:
                    ip += 1
                    break
            except Exception:
                continue
        for i in top_urls:
            try:
                if domain == whois.whois(i).domain:
                    ul += 1
                    break
            except Exception:
                continue
        if ip > 0 or ul > 0:
            data.append(-1)
        else:
            data.append(1)
    except:
        data.append(-1)


def go(url):
    having_ip_address(url)
    long_url(url)
    shorten_url(url)
    at_the_rate_symbol(url)
    double_slash_redirecting(url)
    prefix_suffix(url)
    having_sub_domain(url)
    ssl(url)
    domain_registration_length(url)
    favicon(url)
    port(url)
    https_token(url)
    request_url(url)
    url_of_anchor(url)
    link_in_tag(url)
    sfh(url)
    submitting_to_email(url)
    abnormal_url(url)
    redirect(url)
    on_mouse_over(url)
    right_click(url)
    popup_window(url)
    iframe(url)
    age_of_domain(url)
    dns_record(url)
    web_traffic(url)
    page_rank(url)
    google_index(url)
    links_pointing_to_page(url)
    statistical_report(url)
    
    print(data)
    
    if check(data) == -1:
        return 'PHISHING URL'
    else:
        return 'LEGITIMATE URL'
    
