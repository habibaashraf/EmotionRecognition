class EmotionRecognizer:
    accuracy = 0
    predictions = []

    def arrangeFiles(self, emotion):  #randomly shuffle files and split 80% training-20% testing
        files = glob.glob("dataset//%s//*" % emotion)
        random.shuffle(files)
        training = files[:int(len(files) * 0.8)]
        prediction = files[-int(len(files) * 0.2):]
        return training, prediction

    def getData(self, extraction): #get training and test data with labels
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in emotions:
            print(" working on %s" % emotion)
            training, prediction = self.arrangeFiles(emotion)

            for item in training:
                image = cv2.imread(item)  # open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                clahe_image = clahe.apply(gray)
                extraction.get_landmarks(clahe_image)
                if data['landmarks_vectorised'] == "error":
                    print("no face detected on this one")
                else:
                    training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                    training_labels.append(emotions.index(emotion))

            for item in prediction:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe_image = clahe.apply(gray)
                extraction.get_landmarks(clahe_image)
                if data['landmarks_vectorised'] == "error":
                    print("no face detected on this one")
                else:
                    prediction_data.append(data['landmarks_vectorised'])
                    prediction_labels.append(emotions.index(emotion))

        return training_data, training_labels, prediction_data, prediction_labels

    def runRecognizer(self, clf, classifierName, extraction):
        accur = []
        sum = []
        print("Making sets %s" % i)  # Make sets by random sampling 80%-20%
        training_data, training_labels, prediction_data, prediction_labels = self.getData(extraction)
        npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        print("training ", classifierName, "%s" % i)  # train SVM
        clf.fit(npar_train, training_labels)
        print("getting accuracies %s" % i)  # Use score() function to get accuracy
        npar_pred = np.array(prediction_data)
        pred = clf.score(npar_pred, prediction_labels)
        print (classifierName, "accuracy: ", pred)
        accur.append(pred)  # Store accuracy in a list