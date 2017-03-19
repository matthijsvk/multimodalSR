# coding=utf-8
import urllib, urllib2
import re
import os
import datetime
import threading

g_mutex=threading.Condition()
g_pages=[] #从中解析所有url链接
g_dirs=[] #临时保存所有的dir路径
g_queueURL=[] #等待爬取的url链接列表
g_queueData_dir=[]#等待创建的文件夹列表
g_existdirs=[] #已经爬取过的dir
g_existURL=[] #已经爬取过的url链接列表
g_failedURL=[] #下载失败的url链接列表
g_totalcount=0 #下载过的页面数
starttime=0.0
def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html
class Crawler:
  def __init__(self,crawlername,url,threadnum,datadir):
    self.crawlername=crawlername
    self.url=url
    self.threadnum=threadnum
    self.threadpool=[]
    self.datadir = datadir
    self.logfile=file("log.txt",'w')
  def craw(self):
    global g_queueURL
    #add the TIMIT url to the initial g_queueURL list
    g_queueURL.append(self.url)
    g_queueData_dir.append(self.datadir)
    depth=0
    print self.crawlername+" started..."
    while(len(g_queueURL)!=0):
      depth+=1
      print 'Searching depth ',depth,'...\n\n'
      self.logfile.write("URL:"+g_queueURL[0]+"........")
      #download all this dir's file
      self.downloadAll()

      #update all subdir path to the queueURL list
      self.updateQueueURL()

      content='\n>>>Depth '+str(depth)+':\n'
      self.logfile.write(content)
      i=0
      while i<len(g_queueURL):
        content=str(g_totalcount+i)+'->'+g_queueURL[i]+'\n'
        self.logfile.write(content)
        i+=1
  def downloadAll(self):
    global g_queueURL
    global g_totalcount
    global g_queueData_dir
    i=0
    while i<len(g_queueURL):
      j=0
      while j<self.threadnum and i+j < len(g_queueURL):
        g_totalcount+=1
        #download all files in the g_queueURL[i+j]
        threadresult=self.download(g_queueURL[i+j],g_queueData_dir[i+j],j)
        if threadresult!=None:
          print 'Thread started:',i+j,'--File number =',g_totalcount
        j+=1
      i+=j
      for thread in self.threadpool:
        thread.join(10)
      #don't if this it correct
      self.threadpool=[]
    g_queueURL=[]
    g_queueData_dir=[]
#download all files from url and save to datadir
  def download(self,url,datadir,tid):
    crawthread=CrawlerThread(url,datadir,tid)
    self.threadpool.append(crawthread)
    crawthread.start()
    return 1
#添加新的URL
  def updateQueueURL(self):
    global g_queueURL
    global g_existURL
    global g_queueData_dir
    global g_existdirs
    global g_pages
    global g_dirs
    newUrlList=[]
    newDirList = []
    #add the new dir path to it
    for url in g_pages:
      newUrlList+=url
    #substract the exist crawed URL
    g_queueURL=list(set(newUrlList)-set(g_existURL))
    for dir_path in g_dirs:
        newDirList += dir_path
    g_queueData_dir = list(set(newDirList) - set(g_existdirs))

def getUrl(self, content):
    pass
class CrawlerThread(threading.Thread):
  def __init__(self,url,datadir,tid):
    threading.Thread.__init__(self)
    global g_mutex
    global g_failedURL
    global g_queueURL

    self.url=url
    self.datadir=datadir
    self.tid=tid

    self.url_list = []
    self.dir_list = []
    self.file_list = []

    html = getHtml(self.url)
    reg = r'<a href="([^\"]+)"'
    href_re = re.compile(reg)
    href_list = re.findall(href_re, html)
    href_list=href_list[5:len(href_list)]
    for href_url in href_list:
        if not href_url.startswith("?") and not href_url.endswith("2007/"):
            if href_url.endswith("/"):
                dir_path = self.datadir + os.sep + href_url[0:len(href_url)-1]
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                url = self.url + href_url
                self.url_list.append(url)
                self.dir_list.append(dir_path)
            else:
                self.file_list.append(href_url)
                global g_totalcount
                g_mutex.acquire()
                g_totalcount = g_totalcount + 1
                g_mutex.release()
    g_mutex.acquire()
    g_pages.append(self.url_list)
    g_dirs.append(self.dir_list)
    g_existURL.append(self.url)
    g_existdirs.append(self.datadir)
    g_mutex.release()
  def run(self):
    global g_mutex
    global g_failedURL
    global g_queueURL
    try:
        for href_url in self.file_list:
            data_path = self.datadir + os.sep + href_url
            url = self.url + href_url
            f = urllib2.urlopen(url)
            meta = f.info()
            file_size = int(meta.getheaders("Content-Length")[0])
            file_size_dl = 0
            block_sz = 8192
            file_local = open(data_path, 'wb')
            now_percent_i = 1
            while True:
                buffer = f.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                file_local.write(buffer)
                if file_size_dl / float(file_size) > now_percent_i / 100.0:
                    global starttime
                    endtime = datetime.datetime.now()
                    interval = (endtime - starttime).seconds
                    print "process %d,time collapese:%d s in %s, downloading %s: %f percent" % (self.tid, interval, data_path, url, (file_size_dl / float(file_size)) * 100)
                    now_percent_i = now_percent_i + 1
            file_local.close()
    except Exception,e:
      g_mutex.acquire()
      g_existURL.append(self.url)
      g_failedURL.append(self.url)
      g_mutex.release()
      print 'Failed downloading and saving',self.url
      print e
      return None

if __name__=="__main__":
    baseurl= "http://www.fon.hum.uva.nl/david/ma_ssp/2007/TIMIT/"
    threadnum= 128
    crawlername="TIMIT_Crawler"

    local_data_dir = os.path.abspath('../data')

    global starttime
    starttime = datetime.datetime.now()

    crawler=Crawler(crawlername,baseurl,threadnum,local_data_dir)
    crawler.craw()