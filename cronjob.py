import crawl
import Analyser
from datetime import datetime
import schedule
import time


def now_time():
    now = datetime.now()
    dt_string = now.strftime("%B %d, %Y %H:%M:%S")
    print('----------------------------------------------------------------------------------')
    print("Record Time is", dt_string)
    print('----------------------------------------------------------------------------------')


def job():
    now_time()
    crawl.main()
    Analyser.main()


print('Job Starts')
Analyser.load_model()

schedule.every(5).minutes.do(job)


while True:
    schedule.run_pending()
    time.sleep(1)
