# Handwriting Generation using C-VRNNs

### Intro

For approach 2 we utilised a Conditional Variational Recurrent Neural Network to synthesie Handwriting.

### Reference Repo

For this approach we have taken inspiration from [this repository](https://github.com/emreaksan/deepwriting).

### Steps to Execute the code

1. Clone this repo
   ```
   git clone https://github.com/akshayabali/CS585_Project
   ```

2. cd into the Approach
   ``` 
   cd Approach-2 
   ```

3. Clone deepwriting Git Repo
   ``` 
   git clone https://github.com/emreaksan/deepwriting
   ```

4. Copy the project code into the git repo
   ``` 
   cp ./Project.py ./deepwriting/ 
   ```

5. Download the dataset
   ``` 
   wget -c https://ait.ethz.ch/projects/2018/deepwriting/downloads/deepwriting_dataset.zip 
   ```

6. Extract the zip file 

7. Download and extract the trained model
   ```
   wget -c https://ait.ethz.ch/projects/2018/deepwriting/downloads/tf-1514981744-deepwriting_synthesis_model.tar.gz -O - | tar -xz  
   ```

8. Execute and evaluate the model
   ``` 
   cd deepwriting
   python3 project.py
   ```





