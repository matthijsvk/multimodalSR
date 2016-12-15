# use this to find the number of files in directories (go through the whole tree, but only show the top level directories)
# https://forum.xfce.org/viewtopic.php?id=9106

# count nb of files per directory, print nbFiles and dirName, sort on nbFiles
files=`find . -maxdepth 1 -type d -print0 | while read -d '' -r dir; do num=$(find $dir -ls | wc -l); printf "%5d files in directory %s\n\r" "$num" "$dir"; done | perl -lane 'print "$F[0] $F[-1]\r\n"' | sort -nrk 1`

# show the number of files and the dirname in a pop-up box
echo "$files" | zenity --text-info

# print only last column (the dir name), sav to text file
echo "$files" | perl -lane 'print "$F[-1]"' > dirToRemove.txt

# now, modify text file so that it only contains the folders to remove

# then remove the folders
#cat dirToRemove.txt | xargs -d \\n rm -r




