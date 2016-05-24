package Sentence;

/**
 * Created by monadiab on 4/12/16.
 */
public class PADependencyTuple implements Comparable {

    int predIndex;
    int argIndex;
    String srl;

    PADependencyTuple()
    {
        this.predIndex= -1;
        this.argIndex= -1;
        this.srl= "";
    }
    PADependencyTuple(int predIndex, int argIndex, String srl)
    {
        this.predIndex= predIndex;
        this.argIndex= argIndex;
        this.srl= srl;
    }

    public  int getPredIndex()
    {return predIndex; }

    public  int getArgIndex()
    {return argIndex; }

    public  String getSRL()
    {return srl; }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof PADependencyTuple) {
            PADependencyTuple PADepTuple = (PADependencyTuple) obj;
            if (PADepTuple.predIndex != predIndex)
                return false;
            if (PADepTuple.argIndex != argIndex)
                return false;
            if (!PADepTuple.srl.equals(srl))
                return false;
            return true;
        }
        return false;
    }

    @Override
    public int compareTo(Object o) {
        if (equals(o))
            return 0;
        return hashCode() - o.hashCode();
    }

    @Override
    public int hashCode() {
        int hash = 0;
        hash^= srl.hashCode()* predIndex * argIndex;
        return hash;
    }

}
