package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.util.Pair;

import java.util.*;

/**
 * Represents a partial or complete dependency parse of a sentence, and
 * provides convenience methods for analyzing the parse.
 *
 * @author Danqi Chen
 */
class DependencyTree {

  int n;
  List<Integer> head;
  List<String> label;

  Map<Integer, Map<Integer, Integer>> dists;
  private int counter;

  public DependencyTree() {
    n = 0;
    head = new ArrayList<Integer>();
    //head.add(Config.NONEXIST);
    label = new ArrayList<String>();
    //label.add(Config.UNKNOWN);
  }

  public DependencyTree(DependencyTree tree) {
    n = tree.n;
    head = new ArrayList<Integer>(tree.head);
    label = new ArrayList<String>(tree.label);
  }

  /**
   * Add the next token to the parse.
   *
   * @param h Head of the next token
   * @param l Dependency relation label between this node and its head
   */
  public void add(int h, String l) {
    ++n;
    head.add(h);
    label.add(l);
  }

  public void buildDists(int upper) {
    dists = new HashMap<>();
    for (int i = 0; i < this.n; i++) {
      for (int j = i+1; j < this.n; j++) {
        int currDist = pathLength(i, j, upper);
        if (currDist > upper)
          currDist = upper;

        if (!dists.containsKey(i))
          dists.put(i, new HashMap<>());
        dists.get(i).put(j, currDist);

        if (!dists.containsKey(j))
          dists.put(j, new HashMap<>());
        dists.get(j).put(i, currDist);

        //System.err.println(i + ":"+ j+ ":"+ toks[i]+ ":"+ toks[j]+ ":" + getDepDist(i, j) + ":"+ getDepDist(j, i));
      }
    }
  }

  public int getDepDist(int first, int second) {
    //System.err.println(first + ":" + second);
    return dists.get(first).get(second);
  }

  /**
   * Establish a labeled dependency relation between the two given
   * nodes.
   *
   * @param k Index of the dependent node
   * @param h Index of the head node
   * @param l Label of the dependency relation
   */
  public void set(int k, int h, String l) {
    head.set(k, h);
    label.set(k, l);
  }

  public int getHead(int k) {
    if (k < 0 || k >= n) //Just changed, need further modification. Bug?
      return Config.NONEXIST;
    else
      return head.get(k);
  }

  public String getLabel(int k) { //Bug?
    if (k < 0 || k >= n)
      return Config.NULL;
    else
      return label.get(k);
  }

  //See if the path between the two nodes are within a upper bound
  public int pathLength(int first, int second, int upper) {
    Map<Integer, Integer> firstDists = new HashMap<>();
    int curr = first;
    firstDists.put(curr, 0);

    for (int i = 0; i < upper; i++) {
      if (head.get(curr) == -1)
        break;
      curr = head.get(curr);

      if (curr == second)
        return i+1;
      firstDists.put(curr, i+1);
    }
    curr = second;
    for (int i = 0; i < upper; i++) {
      if (head.get(curr) == -1) //The two nodes do not reach
        return upper;
      curr = head.get(curr);

      if (firstDists.containsKey(curr)) {
        return firstDists.get(curr) + i + 1;
      }
    }

    return upper;
  }

  /**
   * Get the index of the node which is the root of the parse (i.e.,
   * that node which has the ROOT node as its head).
   */
  public int getRoot() {
    for (int k = 0; k < n; ++k)
      if (getHead(k) == -1)
        return k;
    return 0;
  }

  public void printDep(String[] tokSeq) {
    for (int i = 0; i < this.n; i++) {
      int head = getHead(i);
      System.err.println(i + ", head:" + head);
      String currWord = tokSeq[i];
      String headWord = "Root";
      if (head != -1)
          headWord = tokSeq[head];
      System.err.println("Current word:"+ currWord + ", head word:" + headWord);
      for (int j = i+1; j < this.n; j++) {
        String secondWord = tokSeq[j];
        int dist = this.dists.get(i).get(j);
        System.err.println(currWord + " , "+ secondWord + " distance: "+ dist);
      }
    }
  }

  /**
   * Check if this parse has only one root.
   */
  public boolean isSingleRoot() {
    int roots = 0;
    for (int k = 0; k < n; ++k)
      if (getHead(k) == -1)
        roots = roots + 1;
    return (roots == 1);
  }

  // check if the tree is legal, O(n)
  public boolean isTree() {
    List<Integer> h = new ArrayList<Integer>();
    h.add(-1);
    for (int i = 1; i <= n; ++i) {
      if (getHead(i) < 0 || getHead(i) > n)
        return false;
      h.add(-1);
    }
    for (int i = 1; i <= n; ++i) {
      int k = i;
      while (k > 0) {
        if (h.get(k) >= 0 && h.get(k) < i) break;
        if (h.get(k) == i)
          return false;
        h.set(k, i);
        k = getHead(k);
      }
    }
    return true;
  }

  // check if the tree is projective, O(n^2)
  public boolean isProjective() {
    if (!isTree())
      return false;
    counter = -1;
    return visitTree(0);
  }

  // Inner recursive function for checking projectivity of tree
  private boolean visitTree(int w) {
    for (int i = 1; i < w; ++i)
      if (getHead(i) == w && visitTree(i) == false)
        return false;
    counter = counter + 1;
    if (w != counter)
      return false;
    for (int i = w + 1; i <= n; ++i)
      if (getHead(i) == w && visitTree(i) == false)
        return false;
    return true;
  }

  // TODO properly override equals, hashCode?
  public boolean equal(DependencyTree t) {
    if (t.n != n)
      return false;
    for (int i = 1; i <= n; ++i) {
      if (getHead(i) != t.getHead(i))
        return false;
      if (!getLabel(i).equals(t.getLabel(i)))
        return false;
    }
    return true;
  }

  public void print() {
    for (int i = 1; i <= n; ++i)
      System.out.println(i + " " + getHead(i) + " " + getLabel(i));
    System.out.println();
  }

}