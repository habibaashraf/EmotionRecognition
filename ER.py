class EmotionRecognizer:
    accuracy = 0
    predictions = []

    def get_files(self, emotion):  # Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("dataset//%s//*" % emotion)
        random.shuffle(files)
        training = files[:int(len(files) * 0.8)]  # get first 80% of file list
        prediction = files[-int(len(
            files) * 0.2):]  
        return training, prediction

    def make_sets(self, extraction):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in emotions:
            print(" working on %s" % emotion)
            training, prediction = self.get_files(emotion)
            # Append data to training and prediction list, and generate labels 0-7
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

    def plotConfusionMatrix(self, sum):
        labels = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(sum)
        pl.title('Confusion matrix of the classifier')
        ax.set_title('Confusion matrix of the classifier', color='b', rotation='horizontal', x=0.5, y=-0.15)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        pl.xlabel('Predicted')
        pl.ylabel('True')
        pl.show()

    def runRecognizer(self, clf, classifierName, extraction):
        convMatrixAverage = []
        accur = []
        sum = []
        print("Making sets %s" % i)  # Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels = self.make_sets(extraction)
        npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
        npar_trainlabs = np.array(training_labels)
        print("training ", classifierName, "%s" % i)  # train SVM
        clf.fit(npar_train, training_labels)
        print("getting accuracies %s" % i)  # Use score() function to get accuracy
        npar_pred = np.array(prediction_data)
        pred = clf.score(npar_pred, prediction_labels)
        print (classifierName, "accuracy: ", pred)
        accur.append(pred)  # Store accuracy in a list
        trial = clf.predict(npar_pred)
        cm = confusion_matrix(trial, prediction_labels)
        if i == 0:
            sum = cm
        else:
            sum = np.add(sum, cm)
