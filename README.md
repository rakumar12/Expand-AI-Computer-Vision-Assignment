# Expand-AI-Computer-Vision-Assignment

Chest X-Ray images play a vital role in the diagnosis of diseases.

Introduction:
A worldwide health emergency brought on by the COVID-19 pandemic has resulted in an unprecedented need for medical supplies and manpower. Automated technologies, especially in places with few healthcare resources, can aid in the early identification and management of the condition. The creation of a deep learning model that can identify COVID-19 from chest X-ray pictures is one such technique. In this paper, we describe the outcomes of classifying chest X-ray pictures into three groups: normal, pneumonia, and COVID-19 using a deep learning model trained on a collection of images.
The COVID-19 Database, which includes chest X-ray pictures of COVID-19-positive patients, patients with viral pneumonia, and healthy patients, was utilized as the dataset for this study. With 720 photos in the training set and 180 images in the test set, the dataset was split into an 80:20 train-test split.
Background:
The ResNet networks are the most appropriate ones to use when building a neural network adapted to this purpose. We decided to use the Resnet18 network with a three-class data set. Popular pre-trained deep learning model ResNet-18 has demonstrated outstanding performance on a variety of image classification tasks, including medical picture analysis. ResNet-18 is a promising candidate for transfer learning to other image classification problems since it was trained on the ImageNet dataset, which contains a huge number of different pictures.
ResNet-18 can learn to recognize patterns and characteristics in the X-ray pictures that differentiate between normal, pneumonia, and COVID-19 cases when used to diagnose COVID-19 from chest X-ray images. In comparison to bigger models, the model's design is also very straightforward and computationally efficient, making it simpler to train and deploy on a variety of hardware and devices.
Overall, because to its pre-trained weights, superior performance on picture classification tasks, and effectiveness in terms of training and inference time, ResNet-18 is a viable option for this purpose.


Data Acquisition and Pre-processing:
The dataset was obtained from Expand AI Team
For the implementation of this project, the following 3-class dataset was used. A test set with 60 images of each class was set aside, in addition to a train set with 240 images distributed as follows:
 
Before training the model, a contrast stretch was then applied to all the photos in order to improve the visualization of features in low-contrast images. 

Transfer Learning and Model Training:
Once the data acquisition and preprocessing stage were carried out, several transformations were applied to the dataset, such as a Resize, a RandomHorizontalFlip and a normalization. This was done using Pytorch dependencies. Subsequently, based on a transfer learning model, a ResNet18 neural network was implemented starting from a pre-trained model within Pytorch dependencies. By evaluating the predictions made by the model during the training, this process was continued until an accuracy of 0.95 was obtained.

Methodology: 
To divide chest X-ray pictures into three groups, we employed transfer learning and fine-tuning of a pre-trained ResNet-18 model. The last layer was changed to a fully connected layer with three output classes once the pre-trained model had been loaded. The model was subsequently trained using the training set for 10 epochs with a batch size of 16. Adam was the optimizer, and the learning rate was set at 0.001. The model was evaluated on the test set.

Results: 
The model's accuracy on the test set was 96%, with precision, recall, and f1-scores for the NORMAL class being 0.95, 0.95, and 0.95; for the PNEUMONIA class being 0.93, 0.95, and 0.94; and for the COVID class being 1.00, 0.98, and 0.99. The train set's classification report revealed a 97% accuracy rate, with precision, recall, and f1-scores for the NORMAL class of 0.97, 0.97, and 0.97; the PNEUMONIA class of 0.96, 0.96, and 0.96; and the COVID class of 0.98, 0.99, and 0.98. Below is a representation of the classification report from Sklearn

Conclusion:
The deep learning model demonstrated its capacity to identify COVID-19 from chest X-ray pictures by performing well on the test set. It is crucial to keep in mind that the dataset utilized for this study was small, and the model's performance may differ on bigger datasets. By tweaking hyperparameters like the learning rate and the number of epochs, the model may be made even better. To increase the model's accuracy, future studies may potentially include merging additional imaging modalities like CT scans and fusing them with chest X-ray pictures. Overall, the creation of automated methods, such deep learning models, can help with the early detection and treatment of COVID-19 and lessen the strain on medical resources.

Classification report for test set:

              precision    recall   f1-score   support

      NORMAL       0.95      0.95      0.95       245
   PNEUMONIA       0.93      0.95      0.94       223
       COVID       1.00      0.98      0.99       252

    accuracy                           0.96       720
   macro avg       0.96      0.96      0.96       720
weighted avg       0.96      0.96      0.96       720


Classification report for train set:

              precision    recall   f1-score   support

      NORMAL       0.97      0.97      0.97       250
   PNEUMONIA       0.96      0.96      0.96       239
       COVID       0.98      0.99      0.98       231

    accuracy                           0.97       720
   macro avg       0.97      0.97      0.97       720
weighted avg       0.97      0.97      0.97       720


References:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7521412/
https://github.com/ChristianConchari/COVID-19-detection-with-Chest-X-Ray-using-PyTorch
https://ieeexplore.ieee.org/document/9344870
https://www.hindawi.com/journals/jhe/2021/6799202/
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265949
https://github.com/mohd-faizy/09P_Detecting_COVID_19_with_Chest_X-Ray_using_PyTorch
