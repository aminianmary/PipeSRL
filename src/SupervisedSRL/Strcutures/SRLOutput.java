package SupervisedSRL.Strcutures;

/**
 * Created by Maryam Aminian on 2/28/17.
 */
public class SRLOutput {
    String sentence;
    Double confidenceScore;

    public SRLOutput(){
        sentence ="";
        confidenceScore =0.0;
    }
    public SRLOutput(String sen, double score){
        sentence = sen;
        confidenceScore = score;
    }

    public String getSentence() {
        return sentence;
    }

    public Double getConfidenceScore() {
        return confidenceScore;
    }
}
