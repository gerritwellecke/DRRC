import logging as log
import time

start_time = time.time()


loglevel_debug_RC = 11
loglevel_debug_DRRC = 12
loglevel_debug_parallelreservoirs = 13
loglevel_info_single_run = 19
loglevel_info_iterate_run = 20


class ElapsedTimeFormatter(log.Formatter):
    def format(self, record):
        elapsed_time = time.time() - start_time
        record.elapsed_time = f"{elapsed_time:.2f}"
        return super().format(record)


# Create a custom logger
drrc_logger = log.getLogger("drrc_logger")

# Add logger levels
log.addLevelName(loglevel_debug_RC, "DEBUG_RC")
log.addLevelName(loglevel_debug_DRRC, "DEBUG_DRRC")
log.addLevelName(loglevel_debug_parallelreservoirs, "DEBUG_PRC")
log.addLevelName(loglevel_info_single_run, "INFO_1RUN")
log.addLevelName(loglevel_info_iterate_run, "INFO_nRUN")

# Create handlers
c_handler = log.StreamHandler()
c_handler.setLevel(log.NOTSET)

# Create formatters and add it to handlers
c_format = ElapsedTimeFormatter("[%(elapsed_time)9s] %(levelname)s:\t%(message)s")
c_handler.setFormatter(c_format)

# Add handlers to the logger only if there are none
if not drrc_logger.handlers:
    drrc_logger.addHandler(c_handler)

# Set default level of logger
drrc_logger.setLevel(log.INFO)
