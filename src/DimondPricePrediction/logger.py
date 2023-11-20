import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#LOG_FILE="mylog.log"
log_path=os.path.join(os.getcwd(),"logs") ### path + folder

os.makedirs(log_path,exist_ok=True)
#line is used to create a directory (folder) specified by the variable log_path

LOG_FILEPATH=os.path.join(log_path,LOG_FILE) #, combining the folder path (log_path) and the file name (LOG_FILE).

logging.basicConfig(level=logging.INFO, 
                    filename=LOG_FILEPATH,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
                    
)


if __name__ == '__main__': ##he purpose of using this block is to have specific actions or code that should only run when the script is executed directly, not when it's imported as a module. T
    logging.info(" i am tesitng")
##logging.info("here again I am testing") is a log statement using the info level of the logging module. It logs the message "here again I am testing" when the script is run directly.