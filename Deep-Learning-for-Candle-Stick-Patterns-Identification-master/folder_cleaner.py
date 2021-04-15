import os
import shutil

def createIfNotExists(folder_name):
    if os.path.exists(folder_name):
        return True
    else:
        os.makedirs(folder_name)

def placeIntoFolder(i):
    if os.path.splitext(i)[1] in img_ext:
        shutil.move(f'{current_directory}/{i}', f'{current_directory}/Images/{i}')
    if os.path.splitext(i)[1] in video_ext:
        shutil.move(f'{current_directory}/{i}', f'{current_directory}/Videos/{i}')
    if os.path.splitext(i)[1] in doc_ext:
        shutil.move(f'{current_directory}/{i}', f'{current_directory}/Documents/{i}')
    if os.path.splitext(i)[1] in code_ext:
        shutil.move(f'{current_directory}/{i}', f'{current_directory}/Code Files/{i}')
current_directory = os.getcwd()
l = os.listdir(current_directory)

folder_list = ['Images', 'Videos', 'Documents', 'Code Files']
for i in folder_list:
    createIfNotExists(i)
img_ext = ['.png', '.jpg', '.jpeg', '.svg','.PNG', '.jpg', '.JPEG']
doc_ext = ['.pdf', '.odt', '.docx', '.txt']
video_ext = ['.mp4', '.mkv']
code_ext = ['.c', '.py', '.js', '.html']
for i in l:
    placeIntoFolder(i)
