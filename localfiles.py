
'''
collect all text files in given directory with metadata fields
cmd ine
find . -type f -exec file {} \; | grep ":.* ASCII text"

f we want to find all text files in the current directory, including its sub-directories, then we have to augment
the command using the Linux find command:
find . -type f -exec file -i {} \; | grep " text/plain;" | wc

'''