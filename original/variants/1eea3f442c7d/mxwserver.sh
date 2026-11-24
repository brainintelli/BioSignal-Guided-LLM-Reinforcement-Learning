#!/bin/bash

export BIN_PATH="$(realpath "$(dirname "${0}")")"
source ${BIN_PATH}/.env

LOGS_DIR="/tmp/MaxLab/$USER/mxwserver"

if [ ! -d $LOGS_DIR ]; then
  mkdir -p $LOGS_DIR || echo "NOTE: Could not create logs directory"
fi

# Setup correct write permissions
chmod a+w /tmp/MaxLab 2> /dev/null

# Cleanup logs directory
CURRENT_DATE=`date +"%Y%m%d"`
for file in `ls -1 $LOGS_DIR`; do
  FILE_DATE=`basename $file .log`
  if (( (`date -d $CURRENT_DATE +%s` - `date -d $FILE_DATE +%s`) / (24*3600) > 7 )); then
    rm $LOGS_DIR/$file || echo "NOTE: Could not cleanup logs directory"
  fi
done

# Write the current date in the logs file
LOGS_FILE=$LOGS_DIR/$CURRENT_DATE.log
echo "" >> $LOGS_FILE
echo "#############################################################################" >> $LOGS_FILE
date >> $LOGS_FILE
echo "" >> $LOGS_FILE

${BIN_PATH}/mxwserver $@ 2>&1 | tee -a ${LOGS_FILE}
