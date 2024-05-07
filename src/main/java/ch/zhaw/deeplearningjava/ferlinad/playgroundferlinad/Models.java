package ch.zhaw.deeplearningjava.ferlinad.playgroundferlinad;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

public final class Models {

    // number of classification labels: boots, sandals, shoes, slippers
    public static final int NUM_OF_OUTPUT = 4;
    public static final int NUM_OF_LAYERS = 50;

    // height and weight for pre-processing images
    public static final int IMAGE_HEIGHT = 100;
    public static final int IMAGE_WIDTH = 100;

    // name of model
    public static final String MODEL_NAME = "shoeClassifier";

    private Models() {
    }

    public static Model getModel() {
        Model model = Model.newInstance(MODEL_NAME);

        // composable unit of a NN
        Block resNet50 = ResNetV1.builder()
            .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
            .setNumLayers(NUM_OF_LAYERS)
            .setOutSize(NUM_OF_OUTPUT)
            .build();
        
        model.setBlock(resNet50);
        return model;
    }

    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }

}
