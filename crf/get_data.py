from bs4 import BeautifulSoup
import re
from urllib.request import urlopen
from sys import argv
import os

def get_program(programURL, programDir):
    programPage = urlopen(programURL)
    soup = BeautifulSoup(programPage.read())
    courseLink = soup.findAll('li',{'class':'acalog-course'})
    href = [ i.find('a') for i in courseLink ]
    
    anchors = [ re.search('[0-9]{5}[0-9]?', i['onclick']).group() for i in href ]
    
    if not os.path.exists(programDir):
            os.makedirs(programDir)
    
    base1 = 'http://catalogue.usc.edu/ajax/preview_course.php?catoid=7&coid='
    base2 = '&display_options=a:2:{s:8:~location~;s:7:~program~;s:4:~core~;s:5:~34711~;}&show'
    for an in anchors:
        f = urlopen(base1+an+base2)
        content = BeautifulSoup(f.read()).text
        content = content.replace('\n',' ').replace('\r',' ').replace('\t',' ')
        content = content.replace(',',' ,').replace(':',' :').replace('.', ' .')
        content = content.replace('[','[ ').replace(']', ' ] ')
        title = re.search('[A-Z]{2}[A-Z]+ [0-9][0-9]+', content).group()
        with open(programDir+"/"+title+".txt", "w") as text_file:
                text_file.write(content)


if __name__ == '__main__':
    print(argv[1],argv[2])
    get_program(argv[1], argv[2])
