#!/usr/bin/env python    
#coding=utf-8    

import os 
import cv2 
import scipy.io as scio 
import numpy as np

def case_insensitive_sort(liststring):
        listtemp = [(x.lower(),x) for x in liststring]
        listtemp.sort()
        return [x[1] for x in listtemp]
  
class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        
        print("Scan started!")
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            counter = 0
            print(dirpath)
            print(dirnames)
            print(filenames)
            list=[]
            for special_file in filenames:    
                if self.postfix:    
                    special_file.endswith(self.postfix)    
                    files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    special_file.startswith(self.prefix)  
                    files_list.append(os.path.join(dirpath,special_file))    
                else:   
                    counter += 1
                    list.append(os.path.join(dirpath,special_file)) 
                #files_list.append(os.path.join(dirpath,special_file)) 
                    # print(counter)

            if counter > 2:
                files_list.extend(list)
        # print(files_list)
        files_list=case_insensitive_sort(files_list)
        # print("after")
        # print(files_list)   
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list 


name_start = 0
if __name__=="__main__":
    dir="/home/dd/drone_com/picc/images/"  # Will scan the images in read_form in this directory or its subdirectory
    new_path = '/home/dd/drone_com/picc/new_name2/'

    read_form = '.png'
    target_form = '.jpg'
    scan=ScanFile(dir)  
    files=scan.scan_files() 
    files.sort(key= lambda x:int(x.split('/')[-1][:-4]) )  #sort by last num name such as :/home/dd/drone_com/picc/new_name2/105.jpg
    files_num = 0
    for file in files:
        # if os.path.splitext(file)[1] == read_form:
        print(file)
        print(files_num)
        files_num += 1
        img = cv2.imread(file)
        # cv2.imwrite(os.path.splitext(file)[0]+target_form, img)  #splittext:('/home/dd/drone_com/picc/new_name2/105', '.jpg')
        cv2.imwrite(new_path+ str(name_start) +target_form, img)
        name_start += 1
    print("Complete! Files number = ")
    print(files_num)   
        
        
        
        
