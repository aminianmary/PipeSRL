package SupervisedSRL.Strcutures;

/**
 * Created by Maryam Aminian on 2/28/17.
 */
public class SRLOutput {
    String sentence;
    String sentence_w_projected_info;
    Double confidenceScore;

    public SRLOutput(){
        sentence ="";
        sentence_w_projected_info="";
        confidenceScore =0.0;
    }
    public SRLOutput(String sen, String sen2, double score){
        sentence = sen;
        sentence_w_projected_info = sen2;
        confidenceScore = score;
    }

    public String getSentence() {
        return sentence;
    }

    public Double getConfidenceScore() {
        return confidenceScore;
    }

    public String getSentence_w_projected_info() {
        return sentence_w_projected_info;
    }
}
