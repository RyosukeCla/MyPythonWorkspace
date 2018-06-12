from os import environ
from os import path
from dotenv import load_dotenv
import sys
from logging import basicConfig, getLogger, DEBUG

basicConfig(level=DEBUG)
logger = getLogger(__name__)

logger.debug('hello')

dotenv_path = path.join(path.dirname(__file__), '.env')

if path.exists(dotenv_path):
  load_dotenv(dotenv_path)
else:
  logger.warn('didnt load env file. if you want to load private env, put .env to src/config .')

COMETML_API_KEY = environ.get('COMETML_API_KEY')

def __main__():
  print('test init')
