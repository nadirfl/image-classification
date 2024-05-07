package ch.zhaw.deeplearningjava.ferlinad.playgroundferlinad;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.PostMapping;

@RestController
public class ClassificationController {
    
    private Inference inference = new Inference();

    @GetMapping("/ping")
    public String ping() {
        return "Classification app is up and running!";
    }

    @PostMapping("/analyze")
    public String predict(@RequestParam("image") MultipartFile image) throws Exception{
        System.out.println(image);
        return inference.predict(image.getBytes()).toJson();
    }    
}
