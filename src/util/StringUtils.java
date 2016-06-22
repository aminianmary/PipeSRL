package util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.TreeSet;

/**
 * Created by monadiab on 5/17/16.
 */
public class StringUtils {

    // todo
    public static String convertPathArrayIntoString(ArrayList<Integer> depPathArray)
    {
        // todo StringBuilder
        StringBuilder depPath= new StringBuilder();
        for (int dep: depPathArray) {
            depPath.append(dep);
            depPath.append("\t");
        }
        //todo find .replaceAll("\t","_") in all occurrences and remove them!
        return depPath.toString().trim();
    }

    public static String join(Collection<String> collection, String del)
    {
        StringBuilder output= new StringBuilder();
        for (String element: collection) {
            output.append(element);
            output.append("\t");
        }
        //todo find .replaceAll("\t",del) in all occurrences and remove them!
        return output.toString().trim();
    }

}
