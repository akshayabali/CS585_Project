wget http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/words/words.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/ascii/words.txt http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/lines/lines.tgz http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/ascii/lines.txt http://www.fki.inf.unibe.ch/DBs/iamDB/data_processing/xml/xml.tgz --user nnmllab --password datasets

mkdir lines
tar -zxf lines.tgz -C lines
mkdir words
tar -zxf words.tgz -C words
mkdir xml
tar -zxf xml.tgz -C xml

wget http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip
mkdir task
unzip largeWriterIndependentTextLineRecognitionTask.zip -d task
