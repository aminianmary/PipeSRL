package util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.TreeSet;

/**
 * Created by monadiab on 5/17/16.
 */
public class StringUtils {

    // todo
    public static String convertPathArrayIntoString(TreeSet<String> depPathArray)
    {
        // todo StringBuilder
        String depPath= "";
        for (String dep: depPathArray)
            depPath += dep+"\t";
        //todo find .replaceAll("\t","_") in all occurrences and remove them!
        return depPath.trim().replaceAll("\t","_");
    }

    public static String join(Collection<String> collection, String del)
    {
        String output="";
        for (String element: collection)
            output+= element+"\t";
        //todo find .replaceAll("\t",del) in all occurrences and remove them!
        return output.trim().replaceAll("\t",del);
    }

}
