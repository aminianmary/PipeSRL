package Projection;

import SentenceStruct.*;
import SupervisedSRL.Strcutures.ClusterMap;
import SupervisedSRL.Strcutures.IndexMap;
import util.IO;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by monadiab on 3/23/16.
 */
public class Project {

    static int similarSynFunc_tp = 0;
    static int similarSynFunc_fp = 0;
    static int similarSynFunc_fn = 0;

    static int diffSynFunc_tp = 0;
    static int diffSynFunc_fp = 0;
    static int diffSynFunc_fn = 0;

    static int total_tp_predicat = 0;

    static int total_tp_argument = 0;
    static int total_fp_argument = 0;
    static int total_fn_argument = 0;

    static int total_number_of_projected_predicates = 0;
    static int total_number_of_supervised_predicates = 0;

    static int sentences_wo_enough_alignment = 0;

    static HashMap<Integer, HashMap<String, Integer>> depRel_dist_in_projection = new HashMap<Integer, HashMap<String, Integer>>();

    public static void main(String args[]) throws Exception {
        //args[0]: source file with semantic roles in the conll09 format
        //args[1]: target file (each sentence in a separate line)
        //args[2]: source file with Google universal dependencies
        //args[3]: target file with Google universal dependencies
        //args[4]: alignment file
        //args[5]: projected file

        String sourceFile = args[0]; //source file has supervised SRL
        String targetFile = args[1]; //target file has supervised/gold SRL (for comparing purposes)
        String sourceFile_GD = args[2];
        String targetFile_GD = args[3];
        String alignmentFile = args[4];
        String projectedTargetFile = args[5];
        String clusterFilePath = args[6];

        BufferedWriter projectedFileWriter = new BufferedWriter(new FileWriter(projectedTargetFile, true));

        Alignment alignment = new Alignment(alignmentFile);
        HashMap<Integer, HashMap<Integer, Integer>> alignmentDic = alignment.getSourceTargetAlignmentDic();
        HashMap<Integer, HashMap<Integer, Integer>> alignmentDicReverse = alignment.getTargetSourceAlignmentDic();

        final IndexMap indexMap = new IndexMap(sourceFile);
        final ClusterMap clusterMap = new ClusterMap(clusterFilePath);

        ArrayList<String> sourceSents = IO.readCoNLLFile(sourceFile);
        ArrayList<String> targetSents = IO.readCoNLLFile(targetFile);
        ArrayList<String> sourceSents_GD = IO.readCoNLLFile(sourceFile_GD);
        ArrayList<String> targetSents_GD = IO.readCoNLLFile(targetFile_GD);

        //reading source sentence with supervised SRL tags
        for (int idx = 0; idx < sourceSents.size(); idx++) {
            String sourceSent = sourceSents.get(idx);
            String targetSent = targetSents.get(idx);

            if (!sourceSent.startsWith("1\tNULL")) {
                //just if source sentence is not null
                String sourceSent_GD = sourceSents_GD.get(idx);
                String targetSent_GD = targetSents_GD.get(idx);

                boolean decode = false;
                Sentence sourceSentObj = new Sentence(sourceSent, indexMap, clusterMap);
                Sentence targetSentObj = new Sentence(targetSent, indexMap, clusterMap);

                //do projection for sentences with alignment for > 0.9 of source words
                //if (alignmentDic.get(idx).keySet().size() >= (0.9* sourceSentObj.getWords().length)) {
                PAs targetProjectedPAs = projectSRLTagsFromSource2Target(sourceSentObj, targetSentObj,
                        alignmentDic.get(idx), indexMap, projectedFileWriter);

                compareTags(targetProjectedPAs, sourceSentObj, targetSentObj, alignmentDicReverse.get(idx));

           /* }else
            {
                //System.out.println("less than 90% of the words in this sentence do not have alignment: "+idx);
                sentences_wo_enough_alignment++;
            }
            */
            }
        }

        System.out.println("-------------------");

        System.out.println("Sentences without enough (90%) word alignments: " + sentences_wo_enough_alignment);

        System.out.println("-------------------");

        System.out.println("Predicate stats:\ntotal index of projected predicates: " + total_number_of_projected_predicates + "" +
                "\ntotal index of supervised predicates: " + total_number_of_supervised_predicates + "" +
                "\nindex of tp predicates: " + total_tp_predicat + "" +
                "\npredicate precision: " + (double) (total_tp_predicat) / total_number_of_projected_predicates * 100 + "%" +
                "\npredicate recall: " + (double) (total_tp_predicat) / total_number_of_supervised_predicates * 100 + "%\n");

        System.out.println("Argument stats:\ntotal index of tp argument: " + total_tp_argument + "" +
                "\ntotal fp arguments: " + total_fp_argument + "" +
                "\ntotal fn arguments: " + total_fn_argument + "" +
                "\nargument precision: " + (double) (total_tp_argument) / (total_tp_argument + total_fp_argument) * 100 + "%" +
                "\nargument recall: " + (double) (total_tp_argument) / (total_tp_argument + total_fn_argument) * 100 + "%\n");

        System.out.println("-------------------");

        System.out.println("similarSynFunc_tp/total_tp_argumnet: " + (double) similarSynFunc_tp / total_tp_argument * 100 + "%");
        System.out.println("differentSynFunc_tp/total_tp_argument: " + (double) diffSynFunc_tp / total_tp_argument * 100 + "%");

        System.out.println("similarSynFunc_fp/total_fp_argument: " + (double) similarSynFunc_fp / total_fp_argument * 100 + "%");
        System.out.println("differentSynFunc_fp/total_fp_argument: " + (double) diffSynFunc_fp / total_fp_argument * 100 + "%");

        /*
        System.out.println("-----------------------\nSimilar DepRel Distribution:\n\n");

        for (String depRel: depRel_dist_in_projection.keySet())
        {
            System.out.println(depRel+" : tp ("+
                    ((double)depRel_dist_in_projection.get(depRel).get("tp")
                            /(depRel_dist_in_projection.get(depRel).get("tp")+depRel_dist_in_projection.get(depRel).get("fp")))*100 +
                    "%) fp ("+((double)depRel_dist_in_projection.get(depRel).get("fp")/((depRel_dist_in_projection.get(depRel).get("tp")+depRel_dist_in_projection.get(depRel).get("fp"))))*100+"%)");
            System.out.println("--------");
        }
        */

        projectedFileWriter.flush();
        projectedFileWriter.close();

    }


    public static PAs projectSRLTagsFromSource2Target(Sentence sourceSent, Sentence targetSent,
                                                      HashMap<Integer, Integer> sentenceAlignmentDic,
                                                      IndexMap indexMap,
                                                      BufferedWriter projectedFileWriter) throws Exception {
        ArrayList<PA> sourcePredicateArguments = sourceSent.getPredicateArguments().getPredicateArgumentsAsArray();
        ArrayList<PA> projectedPredicateArguments = new ArrayList<PA>();

        HashMap<Integer, String> projectedPIndices = new HashMap<Integer, String>();
        TreeMap<Integer, TreeMap<Integer, String>> projectedArgIndices = new TreeMap<Integer, TreeMap<Integer, String>>();


        int[] sourceWords = sourceSent.getWords();
        int[] sourcePOSTags = sourceSent.getPosTags();

        int[] targetWords = targetSent.getWords();
        int[] targetPOSTags = targetSent.getPosTags();

        for (PA pa : sourcePredicateArguments) {
            int pIdx = pa.getPredicateIndex();
            int pPOS = sourcePOSTags[pIdx];
            //just project verbal predicates to target verbs (as it's the case in German supervised data)

            if (indexMap.int2str(pPOS).startsWith("V")) {
                int pIndex = pa.getPredicateIndex();
                if (sentenceAlignmentDic.containsKey(pIndex)) {

                    int targetPIndex = sentenceAlignmentDic.get(pIndex);
                    int targetPPOS = targetSent.getPosTags()[targetPIndex];
                    int targetPWord = targetSent.getWords()[targetPIndex];

                    if (indexMap.int2str(targetPPOS).contains("V")) {
                        Predicate projectedPred = new Predicate(targetPIndex, pa.getPredicateLabel());
                        projectedPIndices.put(targetPIndex, pa.getPredicateLabel());

                        ArrayList<Argument> args = pa.getArguments();
                        ArrayList<Argument> projectedArgs = new ArrayList<Argument>();

                        for (Argument arg : args) {
                            int aIndex = arg.getIndex();
                            String aType = arg.getType();

                            if (sentenceAlignmentDic.containsKey(aIndex)) {

                                int targetArgIndex = sentenceAlignmentDic.get(aIndex);
                                int targetArgPOS = targetPOSTags[targetArgIndex];

                                Argument projectedArg = new Argument(targetArgIndex, aType);

                                projectedArgs.add(projectedArg);


                                if (!projectedArgIndices.containsKey(targetArgIndex)) {
                                    TreeMap<Integer, String> argInfo = new TreeMap<Integer, String>();
                                    argInfo.put(targetPIndex, aType);
                                    projectedArgIndices.put(targetArgIndex, argInfo);
                                } else {
                                    projectedArgIndices.get(targetArgIndex).put(targetPIndex, aType);
                                }
                            }
                        }
                        projectedPredicateArguments.add(new PA(projectedPred, projectedArgs));
                    }
                }
            }
        }
        PAs projectedPredicateArguments_PAs = new PAs(projectedPredicateArguments);
        return projectedPredicateArguments_PAs;
    }


    public static void writeProjectedRoles(String[] targetWords, HashMap<Integer, String> projectedPIndices,
                                           TreeMap<Integer, TreeMap<Integer, String>> projectedArgIndices,
                                           BufferedWriter projectedFileWriter) throws IOException {
        ArrayList<Integer> targetPIndices = new ArrayList<Integer>(projectedPIndices.keySet());
        Collections.sort(targetPIndices);

        int wordIndex = -1;
        for (String word : targetWords) {
            wordIndex++;
            projectedFileWriter.write(wordIndex + "\t" + word);

            //write projected predicates
            if (targetPIndices.contains(wordIndex))
                projectedFileWriter.write("\t" + projectedPIndices.get(wordIndex));
            else
                projectedFileWriter.write("\t-");

            //write projected arguments
            if (projectedArgIndices.containsKey(wordIndex)) {
                for (int pIndex : targetPIndices) {
                    if (projectedArgIndices.get(wordIndex).containsKey(pIndex))
                        projectedFileWriter.write("\t" + projectedArgIndices.get(wordIndex).get(pIndex));
                    else
                        projectedFileWriter.write("\t-");
                }
            }

            projectedFileWriter.write("\n");
        }

        projectedFileWriter.write("\n");
    }


    public static void compareTags(PAs projectedTags, Sentence sourceSent, Sentence targetSent,
                                   HashMap<Integer, Integer> targetSourceSentenceAlignmentDic) {
        ArrayList<PA> projectedPAs = projectedTags.getPredicateArgumentsAsArray();
        ArrayList<PA> targetPAs = targetSent.getPredicateArguments().getPredicateArgumentsAsArray();

        total_number_of_projected_predicates += projectedPAs.size();
        total_number_of_supervised_predicates += targetPAs.size();

        int[] sourceDepHeads = sourceSent.getDepHeads();
        int[] targetDepHeads = targetSent.getDepHeads();

        int[] sourceDepLabels = sourceSent.getDepLabels();
        int[] targetDepLabels = targetSent.getDepLabels();

        //getting tp rate for predicate (ignoring the true label/just considering index)
        for (PA projectPA : projectedTags.getPredicateArgumentsAsArray()) {
            int projecPI = projectPA.getPredicateIndex();
            String projectPredicateType = projectPA.getPredicateLabel();

            for (PA supervisedPA : targetPAs) {
                if (supervisedPA.getPredicateIndex() == projecPI) //&& supervisedPA.getPredicateLabel().equals(projectPredicateType))
                {
                    //found a correctly projected predicate (tp)
                    total_tp_predicat++;

                    //get rates of argument tp/fp/fn
                    HashSet<PADependencyTuple> projectedPADepTuples = projectPA.getAllPredArgDepTupls();
                    HashSet<PADependencyTuple> supervisedPADepTuples = supervisedPA.getAllPredArgDepTupls();

                    HashSet<PADependencyTuple> tp_argument = new HashSet<PADependencyTuple>(supervisedPADepTuples);
                    HashSet<PADependencyTuple> fp_argument = new HashSet<PADependencyTuple>(projectedPADepTuples);
                    HashSet<PADependencyTuple> fn_argument = new HashSet<PADependencyTuple>(supervisedPADepTuples);

                    tp_argument.retainAll(projectedPADepTuples);
                    fp_argument.removeAll(tp_argument);
                    fn_argument.removeAll(tp_argument);

                    total_tp_argument += tp_argument.size();
                    total_fp_argument += fp_argument.size();
                    total_fn_argument += fn_argument.size();


                    //extract syntactic function
                    for (PADependencyTuple tuple : tp_argument) {
                        int target_pIndex = tuple.getPredIndex();
                        int target_aIndex = tuple.getArgIndex();

                        int source_pIndex = targetSourceSentenceAlignmentDic.get(target_pIndex);
                        int source_aIndex = targetSourceSentenceAlignmentDic.get(target_aIndex);

                        int targetSynFunc = 0;
                        int sourceSynFunc = 0;

                        if (targetDepHeads[target_aIndex] == target_pIndex)
                            targetSynFunc = targetDepLabels[target_aIndex];
                        if (sourceDepHeads[source_aIndex] == source_pIndex)
                            sourceSynFunc = sourceDepLabels[source_aIndex];

                        if (targetSynFunc == sourceSynFunc) {
                            similarSynFunc_tp++;

                            if (!depRel_dist_in_projection.containsKey(targetSynFunc)) {
                                HashMap<String, Integer> tp_fp_map = new HashMap<String, Integer>();
                                tp_fp_map.put("tp", 1);
                                tp_fp_map.put("fp", 0);
                                depRel_dist_in_projection.put(targetSynFunc, tp_fp_map);
                            } else {
                                if (!depRel_dist_in_projection.get(targetSynFunc).containsKey("tp"))
                                    depRel_dist_in_projection.get(targetSynFunc).put("tp", 1);
                                else
                                    depRel_dist_in_projection.get(targetSynFunc).put("tp", depRel_dist_in_projection.get(targetSynFunc).get("tp") + 1);
                            }
                        } else {
                            diffSynFunc_tp++;

                        }

                    }
                    ///////////////////////////////
                    for (PADependencyTuple tuple : fp_argument) {
                        int target_pIndex = tuple.getPredIndex();
                        int target_aIndex = tuple.getArgIndex();

                        int source_pIndex = targetSourceSentenceAlignmentDic.get(target_pIndex);
                        int source_aIndex = targetSourceSentenceAlignmentDic.get(target_aIndex);

                        int targetSynFunc = 0;
                        int sourceSynFunc = 0;

                        if (targetDepHeads[target_aIndex] == target_pIndex)
                            targetSynFunc = targetDepLabels[target_aIndex];
                        if (sourceDepHeads[source_aIndex] == source_pIndex)
                            sourceSynFunc = sourceDepLabels[source_aIndex];

                        if (targetSynFunc == sourceSynFunc) {
                            similarSynFunc_fp++;

                            if (!depRel_dist_in_projection.containsKey(targetSynFunc)) {
                                HashMap<String, Integer> tp_fp_map = new HashMap<String, Integer>();
                                tp_fp_map.put("fp", 1);
                                tp_fp_map.put("tp", 0);

                                depRel_dist_in_projection.put(targetSynFunc, tp_fp_map);
                            } else {
                                if (!depRel_dist_in_projection.get(targetSynFunc).containsKey("fp"))
                                    depRel_dist_in_projection.get(targetSynFunc).put("fp", 1);
                                else
                                    depRel_dist_in_projection.get(targetSynFunc).put("fp", depRel_dist_in_projection.get(targetSynFunc).get("fp") + 1);
                            }

                        } else {
                            diffSynFunc_fp++;

                        }

                    }


                    break;
                }
            }
        }
    }
}
