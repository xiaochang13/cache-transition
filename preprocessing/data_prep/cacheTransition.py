import sys, re

class CacheTransition(object):
  private boolean singleRoot = true;
  public List<String> conceptLabels, arcLabels, transitions;
  public Set<String> unalignLabels;
  public List<String> pushTransitions, conceptIDTransitions, arcTransitions;
  public Map<String, Integer> pushTransitionIDs, conIDTransitionIDs, arcTransitionIDs;
  public Set<Integer> neChoices;
  public Set<Integer> constArcOutChoices;
  public Set<Integer> constArcInChoices;
  //public Map<Integer, Map<String, Integer>> conMap;
  public int cacheSize = 0;

  public CacheTransition(List<String> clabels, List<String>alabels, Set<String>uset, int size) {
    //super(tlp, labels, verbose);
    conceptLabels = clabels;
    unalignLabels = uset;
    arcLabels = alabels;
    //conMap = cmap;

    pushTransitions = new ArrayList<>();
    conceptIDTransitions = new ArrayList<>();
    arcTransitions = new ArrayList<>();

    pushTransitionIDs = new HashMap<>();
    conIDTransitionIDs = new HashMap<>();
    arcTransitionIDs = new HashMap<>();

    cacheSize = size;

  }

  public int numTransitions(int type) {
    if (type == 0) {
      return conceptIDTransitions.size();
    }
    else if (type == 1) {
      return arcTransitions.size();
    }
    return pushTransitions.size();
  }

  public List<String> transitionList(int type) {
    if (type == 0)
      return conceptIDTransitions;
    else if (type == 1)
      return arcTransitions;
    return pushTransitions;
  }

  public String getTransition(int type, int index) {
    if (type == 0)
      return conceptIDTransitions.get(index);
    else if (type == 1)
      return arcTransitions.get(index);
    return pushTransitions.get(index);
  }

  public boolean isTerminal(AMRConfiguration c) {
    return (c.getStackSize() == 0 && c.getBufferSize() == 0);
  }

  public int conIDUNKNOW() {
    return conIDTransitionIDs.get("conID:"+Config.UNKNOWN);
  }

  //Need further modification to make sure
  public Map<Integer, Set<Integer>> makeTransitions(Map<Integer, Map<String, Integer>> conMap) {

    Map<Integer, Set<Integer>> dict = new HashMap<>();
    neChoices = new HashSet<>();
    constArcInChoices = new HashSet<>();
    constArcOutChoices = new HashSet<>();

    for (String label: conceptLabels) { //-NULL- for unaligned words, -UNK- for unknown words
      String currTran = "conID:" + label;
      conIDTransitionIDs.put(currTran, conceptIDTransitions.size());
      conceptIDTransitions.add(currTran);
      if (label.equals("NE") || label.contains("NE_")) {
        neChoices.add(conIDTransitionIDs.get(currTran));
      }
    }

    String nullTran = "conID:" + Config.NULL;
    if (!conIDTransitionIDs.containsKey(nullTran)) {
      conIDTransitionIDs.put(nullTran, conceptIDTransitions.size());
      conceptIDTransitions.add(nullTran);
    }

    String unkTran = "conID:" + Config.UNKNOWN;
    if (!conIDTransitionIDs.containsKey(unkTran)) {
      conIDTransitionIDs.put(unkTran, conceptIDTransitions.size());
      conceptIDTransitions.add(unkTran);
    }

    //for (String label: unalignLabels) {
    //  String currTran = "conGen:" + label;
    //  conIDTransitionIDs.put(currTran, conceptIDTransitions.size());
    //  conceptIDTransitions.add(currTran);
    //}

    for (int wordID: conMap.keySet()) {
      Map<String, Integer> conceptCounts = conMap.get(wordID);
      dict.put(wordID, new HashSet<>());
      for (String s: conceptCounts.keySet()) {
        String action = "conID:" + s;
        dict.get(wordID).add(conIDTransitionIDs.get(action));
      }
    }

    //Unit decision at each vertex of the cache, disallow NULL or UNKNOWN
    for (String label : arcLabels) {
      if (label.equals(Config.NULL) || label.equals(Config.UNKNOWN)) {
        continue;
      }
      String currTran = "L-" + label;
      arcTransitionIDs.put(currTran, arcTransitions.size());
      arcTransitions.add(currTran);

      constArcOutChoices.add(arcTransitionIDs.get(currTran));
    }
    for (String label : arcLabels) {
      if (label.equals(Config.NULL) || label.equals(Config.UNKNOWN)) {
        continue;
      }
      String currTran = "R-" + label;
      arcTransitionIDs.put(currTran, arcTransitions.size());
      arcTransitions.add(currTran);

      constArcInChoices.add(arcTransitionIDs.get(currTran));
    }

    //Finally the push to cache action

    for (int i = 0; i < cacheSize; i++) {
      String pushAction = "PUSH:" + i;
      pushTransitionIDs.put(pushAction, pushTransitions.size());
      pushTransitions.add(pushAction);
    }

    System.err.println(conceptIDTransitions);
    System.err.println(arcTransitions);
    System.err.println(pushTransitions);
    System.err.println("Concept ID:" + conceptIDTransitions.size() + " ; "+ conIDTransitionIDs.size());
    System.err.println("arc Connect:" + arcTransitions.size() + " ; "+ arcTransitionIDs.size());
    System.err.println("PUSH index:" + pushTransitions.size() + " ; "+ pushTransitionIDs.size());

    return dict;
  }

  public int getTransitionID(String action, int type) {
    if (type == 0)
      return conIDTransitionIDs.get(action);
    else if (type == 1)
      return arcTransitionIDs.get(action);
    else if (type == 2)
      return pushTransitionIDs.get(action);
    return -1;
  }

  //Need to fix how to initialize the AMR graph
  public AMRConfiguration initialConfiguration(int[] toks) {
    //AMRConfiguration c = new AMRConfiguration(toks);
    //int length = toks.size();

    //// For each token, add dummy elements to the configuration's tree
    //// and add the words onto the buffer
    //for (int i = 1; i <= length; ++i) {
    //  //c.graph.add(Config.NONEXIST, Config.UNKNOWN);
    //  c.buffer.add(i);
    //}

    //// Put the ROOT node on the stack
    //c.stack.add(0);

    return null;
  }

  //Whether it's possible to apply the next transtions
  public boolean canApply(AMRConfiguration c, String t) {
    //if (t.startsWith("L") || t.startsWith("R")) {
    //  String label = t.substring(2, t.length() - 1);
    //  int h = t.startsWith("L") ? c.getStack(0) : c.getStack(1);
    //  if (h < 0) return false;
    //  if (h == 0 && !label.equals(rootLabel)) return false;
    //  //if (h > 0 && label.equals(rootLabel)) return false;
    //}

    //int nStack = c.getStackSize();
    //int nBuffer = c.getBufferSize();

    //if (t.startsWith("L"))
    //  return nStack > 2;
    //else if (t.startsWith("R")) {
    //  if (singleRoot)
    //    return (nStack > 2) || (nStack == 2 && nBuffer == 0);
    //  else
    //    return nStack >= 2;
    //} else
    //  return nBuffer > 0;
    return true;
  }

  //Apply different types of transitions
  public void apply(AMRConfiguration c, String t) {
    if (t.equals("POP")) {//push or pop
      if (!c.pop()) {
        System.err.println("Pop from empty stack!:(");
        c.printConfig();
        System.exit(1);
      }
      c.startAction = true;
    }
    else if (t.equals("conID:-NULL-")) {
      c.popBuffer();
      c.startAction = true;
    }
    else if (t.contains("conGen") || t.contains("conID")) { //concept identification
      ConceptLabel concept = new ConceptLabel(t.split(":")[1]);
      c.lastType = c.graph.nextConceptId();
      c.lastP = new Pair<>();
      c.lastP.second = c.lastType;
      c.graph.add(concept);
      c.graph.count();
      c.startAction = false; //needs further processing
      if (t.contains("conGen") && c.graph.concepts.size() < c.goldGraph.concepts.size())
          c.startAction = true;
      if (t.contains("conID")) {
        c.lastP.first = c.buffer.get(0);
        c.popBuffer();
      }
      else {
        c.lastP.first = -1;
      }
      c.lastAction = "conID";
    }
    else if (t.contains("ARC")) {//connecting edges between current edge and all the vertices in the cache
      String[] parts = t.split(":");
      int cacheIndex = Integer.parseInt(parts[0].substring(3));
      String currArc = parts[1];
      //String[] arcs = parts[1].split("#");
      c.lastAction = "ARC";

      //Pair<Integer, Integer> p = c.lastP;
      //int currConcept = c.lastP.second;
      //int currWordIndex = c.lastP.first;
      //for (int i = 0; i < c.cacheSize; i++) {
      int cacheVertex = c.cache.get(cacheIndex).second;
      //String currArc = arcs[i];
      if (currArc.equals("O"))
        return;
        else if (currArc.charAt(0) == 'L')
          c.connect(c.lastType, cacheVertex, 0, currArc.substring(2));
        else
          c.connect(c.lastType, cacheVertex, 1, currArc.substring(2));
      //}
      c.startAction = false;
    }
    else {
      int cacheIndex = Integer.parseInt(t.split(":")[1]);
      //c.popBuffer();
      c.pushStack(cacheIndex);
      c.cache.add(c.lastP);
      c.startAction = true;
    }
  }

  public int chooseVertex(AMRConfiguration c, Map<Integer, Set<Integer>> headToTail,
                          Map<Integer, Set<Integer>> tailToHead, Map<Integer, Integer> wordToConcept) {
    int maxDist = -1;
    int maxIndex = -1;
    for (int cachePos = 0; cachePos < c.cacheSize; cachePos++) {
      int cacheVertex = c.cache.get(cachePos).second;
      int currDist = 1000;
      if (cacheVertex == -1) {
        return cachePos;
      }
      Set<Integer> tailSet = headToTail.get(cacheVertex);
      Set<Integer> headSet = null;
      if (tailToHead.containsKey(cacheVertex))
        headSet = tailToHead.get(cacheVertex);
      //Set<Integer> headSet = tailToHead.get(cacheVer)
      for (int bufferPos = 0; bufferPos < c.buffer.size(); bufferPos++) {
        int wordIndex = c.buffer.get(bufferPos);
        if (wordToConcept.containsKey(wordIndex)) {
          int buffIndex = wordToConcept.get(wordIndex);
          if (tailSet.contains(buffIndex) || (headSet != null && headSet.contains(buffIndex))) {
            currDist = bufferPos;
            break;
          }
        }
      }
      if (currDist > maxDist) {
        maxIndex = cachePos;
        maxDist = currDist;
      }
    }
    return maxIndex;
  }

  // O(n) implementation
  public String getOracle(AMRConfiguration c, int wordIndex) {
    AMRGraph graph = c.goldGraph;
    Map<Integer, Set<Integer>> headToTail = graph.headToTail;
    Map<Integer, Set<Integer>> tailToHead = graph.tailToHead;
    Map<Integer, Integer> wordToConcept = graph.wordToConcept;
    String action = null;
    if (c.startAction && c.needsPop()) {
      c.lastAction = "POP";
      action = "POP";
    }
    else if (c.startAction) { //First procedure to process a word index
      // identification of current word
      int conceptIndex = -1;

      c.lastAction = "conID";
      //System.err.println("Current concept:" + graph.concepts.get())
      if (graph.concepts.size() > c.graph.nextConceptId() && (!graph.nextIsAligned(c.graph.nextConceptId()))) {
        //System.err.println("Get here");
        conceptIndex = c.graph.nextConceptId();
        //System.err.println(graph.concepts.size());
        //System.
        //c.printConfig();
        action = "conGen:" + graph.concepts.get(conceptIndex).value();
        //System.exit(1);
      }
      else if (wordToConcept.containsKey(wordIndex)) {
        conceptIndex = wordToConcept.get(wordIndex);
        action = "conID:" + graph.concepts.get(conceptIndex).value();
        if (!conIDTransitionIDs.containsKey(action)) {
          //System.err.println("Unseen:" + action);
          action = "conID:" + Config.UNKNOWN;
        }
      }
      else {
        action = "conID:-NULL-";
        c.lastAction = "emp";
      }
    }
    else if (c.lastAction.equals("conID")) { //processing information involving concept
      int currConcept = c.lastType;
      action = "ARC:";
      List<String> arcs = new ArrayList<>();
      c.lastAction = "ARC";
      for (int i = 0; i < c.cacheSize; i++) {
        int cacheConcept = c.getCache(i).second;
        if (cacheConcept == -1) {
          arcs.add("O");
          continue;
        }

        if (headToTail.get(currConcept).contains(cacheConcept)) {
          String arcL = graph.concepts.get(currConcept).getRel(cacheConcept);
          //String currC =graph.concepts.get(currConcept).value();
          //String cacheC =graph.concepts.get(cacheConcept).value();


          String tmp = "L-" + arcL;
          //System.err.println(cacheC + ":" + tmp + ":" + currC);
          if (!arcTransitionIDs.containsKey(tmp)) {
            System.err.println("Unseen:" + tmp);
            tmp = "L-" + Config.UNKNOWN;
          }
          arcs.add(tmp);
        } else if (headToTail.get(cacheConcept).contains(currConcept)) {
          String arcL = graph.concepts.get(cacheConcept).getRel(currConcept);
          String tmp = "R-" + arcL;
          if (!arcTransitionIDs.containsKey(tmp)) {
            //System.err.println("Unseen:" + tmp);
            tmp = "R-" + Config.UNKNOWN;
          }
          arcs.add(tmp);
        } else {
          arcs.add("O");
        }
      }
      action += StringUtils.join(arcs, "#");
    }
    else if (c.lastAction.contains("ARC")) { //connecting arcs among all cache vertices
      c.lastAction = "PUSH";

      //First choose a position in the cache to push concept to and push that vertex in to stack
      int cacheIndex = chooseVertex(c, headToTail, tailToHead, wordToConcept);
      //c.pushStack(cacheIndex);
      action = "PUSH:" + (cacheIndex);
    }
    else {
      System.err.println("Something wrong here! Action:" + c.lastAction);
      System.exit(1);
    }
    return action;
  }

  // NOTE: unused. need to check the correctness again.
  public boolean canReach(AMRConfiguration c, DependencyTree dTree) {
    //int n = c.getSentenceSize();
    //for (int i = 1; i <= n; ++i)
    //  if (c.getHead(i) != Config.NONEXIST && c.getHead(i) != dTree.getHead(i))
    //    return false;

    //boolean[] inBuffer = new boolean[n + 1];
    //boolean[] depInList = new boolean[n + 1];

    //int[] leftL = new int[n + 2];
    //int[] rightL = new int[n + 2];

    //for (int i = 0; i < c.getBufferSize(); ++i)
    //  inBuffer[c.buffer.get(i)] = true;

    //int nLeft = c.getStackSize();
    //for (int i = 0; i < nLeft; ++i) {
    //  int x = c.stack.get(i);
    //  leftL[nLeft - i] = x;
    //  if (x > 0) depInList[dTree.getHead(x)] = true;
    //}

    //int nRight = 1;
    //rightL[nRight] = leftL[1];
    //for (int i = 0; i < c.getBufferSize(); ++i) {
    //  boolean inList = false;
    //  int x = c.buffer.get(i);
    //  if (!inBuffer[dTree.getHead(x)] || depInList[x]) {
    //    rightL[++nRight] = x;
    //    depInList[dTree.getHead(x)] = true;
    //  }
    //}

    //int[][] g = new int[nLeft + 1][nRight + 1];
    //for (int i = 1; i <= nLeft; ++i)
    //  for (int j = 1; j <= nRight; ++j)
    //    g[i][j] = -1;

    //g[1][1] = leftL[1];
    //for (int i = 1; i <= nLeft; ++i)
    //  for (int j = 1; j <= nRight; ++j)
    //    if (g[i][j] != -1) {
    //      int x = g[i][j];
    //      if (j < nRight && dTree.getHead(rightL[j + 1]) == x) g[i][j + 1] = x;
    //      if (j < nRight && dTree.getHead(x) == rightL[j + 1]) g[i][j + 1] = rightL[j + 1];
    //      if (i < nLeft && dTree.getHead(leftL[i + 1]) == x) g[i + 1][j] = x;
    //      if (i < nLeft && dTree.getHead(x) == leftL[i + 1]) g[i + 1][j] = leftL[i + 1];
    //    }
    //return g[nLeft][nRight] != -1;
    return true;
  }

  public boolean isOracle(AMRConfiguration c, String t, DependencyTree dTree) {
    //if (!canApply(c, t))
    //  return false;

    //if (t.startsWith("L") && !dTree.getLabel(c.getStack(1)).equals(t.substring(2, t.length() - 1)))
    //  return false;

    //if (t.startsWith("R") && !dTree.getLabel(c.getStack(0)).equals(t.substring(2, t.length() - 1)))
    //  return false;

    //Configuration ct = new Configuration(c);
    //apply(ct, t);
    //return canReach(ct, dTree);
    return true;
  }
}
