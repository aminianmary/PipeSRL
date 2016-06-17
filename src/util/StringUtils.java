package util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.TreeSet;

/**
 * Created by monadiab on 5/17/16.
 */
public class StringUtils {

    public static String convertPathArrayIntoString(TreeSet<String> depPathArray)
    {
        String depPath= "";
        for (String dep: depPathArray)
            depPath += dep+"\t";
        return depPath.trim().replaceAll("\t","_");
    }

    public static String join(Collection<String> collection, String del)
    {
        String output="";
        for (String element: collection)
            output+= element+"\t";
        return output.trim().replaceAll("\t",del);
    }

}
