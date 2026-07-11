with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_code = '''    else:
        if ALL_TAB_SCAN_STATUS.get("bb_squeeze_running", False):
            st.info("? Background scanner is analyzing BB Squeezes across Daily, Weekly, and Monthly timeframes... Please wait (~2 minutes).")
            if st.button("?? Refresh BB Squeeze Status", key="refresh_bb_none_btn"):
                st.rerun()
        else:
            st.warning("?? Scan has not been run yet.'''

new_code = '''    else:
        if st.session_state.get("bb_squeeze_running", False) or ALL_TAB_SCAN_STATUS.get("bb_squeeze_running", False):
            st.info("? Background scanner is analyzing BB Squeezes across Daily, Weekly, and Monthly timeframes... Please wait (~2 minutes).")
            if st.button("?? Refresh BB Squeeze Status", key="refresh_bb_none_btn"):
                st.rerun()
        else:
            st.warning("?? Scan has not been run yet.'''

if old_code in code:
    code = code.replace(old_code, new_code)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Replaced successfully')
else:
    print('Code not found')
