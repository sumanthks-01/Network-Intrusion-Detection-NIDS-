from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

scheduler = BackgroundScheduler()

def start_scheduler():
    def heartbeat():
        logger.info("Scheduler Heartbeat OK")

    scheduler.add_job(heartbeat, "interval", minutes=5)
    scheduler.start()
