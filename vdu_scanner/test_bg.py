import sys
import app
import time
print('Initial bb_squeeze_running:', app.ALL_TAB_SCAN_STATUS.get('bb_squeeze_running'))
app.run_background_bb_squeeze_scan(force=True)
print('After function call:', app.ALL_TAB_SCAN_STATUS.get('bb_squeeze_running'))
time.sleep(2)
print('After sleep:', app.ALL_TAB_SCAN_STATUS.get('bb_squeeze_running'))
