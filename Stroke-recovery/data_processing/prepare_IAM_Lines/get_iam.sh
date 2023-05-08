read -p "IAM username: " iam_username
wget http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/lines/lines.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/ascii/lines.txt http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/xml/xml.tgz --user $iam_username --ask-password

mkdir lines
tar -zxf lines.tgz -C lines
mkdir xml
tar -zxf xml.tgz -C xml

wget http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
mkdir task
unzip largeWriterIndependentTextLineRecognitionTask.zip -d task
python main.py
