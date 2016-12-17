#!/bin/bash

# This script downloads the TCD-TIMIT database to the current directory, creating a proper directory structure
# HOW TO USE:
# 1. login to the TCD- TIMIT website
# 2. download the firefox plugin 'cliget'
# 3. go to some file on the TCD- TIMIT downloads page, right click, and 'copy curl for link' in the cliget menu
# 4. get the 'Cookie' header, and replace the link in this script with it
cookieHeader='Cookie: has_js=1; SSESSa08f1a9d41786c65667603b759c65eb0=Kfnjpyh3YTufGYIBWcHhIQAVVFhQIR1JrVqa-_a_sp8'

topDir="$PWD"
echo "Downloading the TCD- TIMIT database to $topDir"
echo "############################################################################"

# read the input file, formatted as: 
# dirName: URL
# so for example '01M: https://sigmedia.tcd.ie/TCDTIMIT/filebrowser/download/1223'

# -> split each line at the colon. The part before = dir name. The part after= URL
# create the directory if it doesn't exist yet, then go down into it and download the file referenced by the URL.
# loop through all the lines of the input file until everything is downloaded


# 1. create top-level directories
mkdir -p lipspeakers
mkdir -p volunteers


IFS=' ' read -ra line <<< `cat $1`  

for i in "${line[@]}"; do
	if [[ $i =~ .*https* ]]; then  # for the URL, you should be cd'ed into the right place. Now download the file.
   		echo "URL: $i"
		if [[ -f "straightcam.zip" ]] || [[ -d "straightcam" ]]; then
    			echo "File exists. Skipping..."
		else
			echo "File does not exist yet. Downloading..."
			# This command comes from the firefox plugin 'cliget' that told me the curl command when downloading from. I removed all unnecessary headers for clarity
			curl --header "$cookieHeader" "$i" -O -J -L
		fi
		cd "$topDir"
		echo "$i downloaded"
		echo "------------------------------------------"

	else	# for the directory, create dir structure and cd into it. Next loop iteration, you'll encounter the URL and download the file over there

		i=`echo "$i" | sed 's/\://g'` #remove colon from dirname
		echo "Folder name: $i"
			
		# make the directories and go in there to then be able to download the file
		if [[ $i =~ .*Lipspkr* ]]; then  # if lipspeaker, go to the lipspeaker top-level folder, otherwise to the volunteer folder
			mkdir -p "$topDir/lipspeakers/$i/Clips"
			cd "$topDir/lipspeakers/$i/Clips"
		else
			mkdir -p "$topDir/volunteers/$i/Clips"
			cd "$topDir/volunteers/$i/Clips"
		fi
	fi
done

curl --header  --header 'DNT: 1' 'https://sigmedia.tcd.ie/TCDTIMIT/filebrowser/download/588' -O -J -L
