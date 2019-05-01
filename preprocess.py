import os,sys
import re
from pydub import AudioSegment
root_directory ="/home/sirena/sumuk/mfcc/youtube/speech_data"
name ="data_sub"
data_dir = os.path.join(root_directory, name)
file_wav=[]
dict_time={}
str_line=""
first_time=0
prev_j=""
save_file="indian_"
#00:00:16.930 --> 00:00:22.390
p = re.compile('^[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9][0-9]*')
for subdir, dirs, files in os.walk(data_dir):
        for f in files:
		if f.endswith('.wav'):
			file_wav.append(os.path.join(subdir, f))
for i in file_wav:
	file_name=os.path.splitext(i)[0]
	txt_name=file_name+".en.vtt"
	f_open=open(txt_name,"r")
	lines=f_open.read().splitlines()
	for j in lines:
		if(p.search(j) != None):
			if(first_time==0):
				first_time+=1
			else:
				dict_time[prev_j]=(str_line,i)
				#print(prev_j,str_line)
			str_line=""
			prev_j=j
		else:
			if(first_time==0):
				continue
			str_line+=" "+j
file_no=7764
save_wav="/home/sirena/sumuk/mfcc/youtube/pro_data/wav"
save_text="/home/sirena/sumuk/mfcc/youtube/pro_data/txt"
zero_fill=len(str(len(dict_time)))
#print(len(dict_time),zero_fill)
for k,v in dict_time.iteritems():
	start_time=k.split("-->")[0]
	end_time=k.split("-->")[1]
	time=start_time.split()[0].split(":")
	start_time_mill=int(time[0])*3600*1000+int(time[1])*60*1000+int(time[-1].split(".")[0])*1000+int(time[-1].split(".")[1])
	#print(start_time,start_time_mill)
	time=end_time.split()[0].split(":")
	end_time_mill=int(time[0])*3600*1000+int(time[1])*60*1000+int(time[-1].split(".")[0])*1000+int(time[-1].split(".")[1])
	if(end_time_mill-start_time_mill>15000):
		print(v,file_no)
	print(start_time_mill,end_time_mill,start_time,end_time,v[0])
    	newAudio = AudioSegment.from_wav(v[1])
    	newAudio = newAudio[start_time_mill:end_time_mill]
    	newAudio.export(os.path.join(save_wav,save_file+str(file_no).zfill(zero_fill)+".wav"), format="wav")
	f_txt=open(os.path.join(save_text,save_file+str(file_no).zfill(zero_fill)+".txt"),"w")
	f_txt.write(v[0])
	f_txt.close
	file_no+=1#'''
