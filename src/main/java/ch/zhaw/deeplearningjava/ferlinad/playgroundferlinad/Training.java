package ch.zhaw.deeplearningjava.ferlinad.playgroundferlinad;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

public final class Training {

    // number of training samples processed before the model is updated
    private static final int BATCH_SIZE = 32;

    // number of passes over the complete dataset
    private static final int EPOCHS = 2;

    public static void main(String[] args) throws IOException, TranslateException {
        Path modelDir = Paths.get("models");

        // alternative: ut-zap50k-images-square-small
        ImageFolder dataset = initDataset("ut-zap50k-images-square");
        // training validation split
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);
        Dataset training = datasets[0];
        Dataset validation = datasets[1];

        // set loss function
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // hyperparameters for config
        TrainingConfig config = setupTrainingConfig(loss);

        Model model = Models.getModel();
        // report KPI's like accuracy
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());

        // initialize trainer with proper input shape (100x100)
        Shape shape = new Shape(1, 3, Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT);
        trainer.initialize(shape);

        // find patterns in data
        EasyTrain.fit(trainer, EPOCHS, training, validation);

        // model properties
        TrainingResult result = trainer.getTrainingResult();
        model.setProperty("Epoch", String.valueOf(EPOCHS));
        model.setProperty(
                "Accuracy",
                String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

        model.save(modelDir, Models.MODEL_NAME);
        // save lavels
        Models.saveSynset(modelDir, dataset.getSynset());
    }

    private static ImageFolder initDataset(String datasetRoot) throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
            .setRepositoryPath(Paths.get(datasetRoot))
            .optMaxDepth(10)
            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
            .addTransform(new ToTensor())
            .setSampling(BATCH_SIZE, true)
            .build();
        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
            .addEvaluator(new Accuracy())
            .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}
