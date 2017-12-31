package edu.stanford.nlp.parser.nndep;

import edu.stanford.nlp.international.Language;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.*;

import static java.util.stream.Collectors.toList;

/**
 * This class defines a transition-based dependency parser which makes
 * use of a classifier powered by a neural network. The neural network
 * accepts distributed representation inputs: dense, continuous
 * representations of words, their part of speech tags, and the labels
 * which connect words in a partial dependency parse.
 *
 * <p>
 * This is an implementation of the method described in
 *
 * <blockquote>
 *   Danqi Chen and Christopher Manning. A Fast and Accurate Dependency
 *   Parser Using Neural Networks. In EMNLP 2014.
 * </blockquote>
 *
 * <p>
 *
 * <p>
 * This parser can also be used programmatically. The easiest way to
 * prepare the parser with a pre-trained model is to call
 * parser instance in order to get new parses.
 *
 * @author Xiaochang Peng
 */
public class AMRParser {
  public static final String DEFAULT_MODEL = "edu/stanford/nlp/models/parser/nndep/english_UD.gz";

  /**
   * Words, parts of speech, and dependency relation labels which were
   * observed in our corpus / stored in the model
   *
   */
  private List<String> knownWords, knownLemmas, knownPos, knownDeps, knownConcepts, knownArcs;
  private List<String> allLabels;

  private Writer pushPopFeatWriter;
  private Writer conceptIDFeatWriter;
  private Writer arcBinaryFeatWriter;
  private Writer arcLabelFeatWriter;
  private Writer pushIndexFeatWriter;

  private Writer outputWriter;

  /**
   * Mapping from word / POS / dependency relation label to integer ID
   */
  private Map<String, Integer> wordIDs, lemIDs, posIDs, depIDs, conceptIDs, arcIDs;
  private Map<String, Integer> conceptIDTargetIDs;
  private Map<Integer, Set<Integer>> wordToConcepts;
  private Map<Integer, Map<String, Integer>> wordConceptCounts;
  private Map<String, Map<String, Integer>> lemmaConceptCounts;
  private Map<Integer, String> mleConceptID;
  private Map<String, String> lemMLEConceptID;
  private Map<Integer, Set<Integer>> conceptIDDict;
  private List<String> unalignedSet;
  public Set<String> constSet;
  public Map<Integer, Integer> connectWordDistToCount;
  public Map<Integer, Integer> nonConnectDistToCount;

  public Map<String, Set<Integer>> word2choices;
  public Map<String, Set<Integer>> lemma2choices;

  public Map<String, Set<String>> concept2outgo;
  public Map<String, Set<String>> concept2income;

  public Set<String> outputActionChoices;
  public Set<Integer> arcLabelDefaultCandidates;

  public Map<Integer, Integer> depConnectDistToCount;
  public Map<Integer, Integer> depNonConnectDistToCount;

  public Map<String, Integer> conceptIDFeatIndexer;
  public Map<String, Integer> conceptIDLabelIndexer;
  public List<String> conceptIDLabels;
  public Map<Integer, Map<Integer, Integer>>conceptIDEmissionScores;

  public Map<String, Integer> arcBinaryFeatIndexer;
  public Map<String, Integer> arcBinaryLabelIndexer;
  public List<String> arcBinaryLabels;
  public Map<Integer, Map<Integer, Integer>>arcBinaryEmissionScores;

  public Map<String, Integer> arcLabelFeatIndexer;
  public Map<String, Integer> arcLabelLabelIndexer;
  public List<String> arcLabelLabels;
  public Map<Integer, Map<Integer, Integer>>arcLabelEmissionScores;

  public Map<String, Integer> pushPopFeatIndexer;
  public Map<String, Integer> pushPopLabelIndexer;
  public List<String> pushPopLabels;
  public Map<Integer, Map<Integer, Integer>>pushPopEmissionScores;

  public Map<String, Integer> pushIndexFeatIndexer;
  public Map<String, Integer> pushIndexLabelIndexer;
  public List<String> pushIndexLabels;
  public Map<Integer, Map<Integer, Integer>>pushIndexEmissionScores;

  /**
   * Given a particular parser configuration, this classifier will
   * predict the best transition to make next.
   *
   * The {@link edu.stanford.nlp.parser.nndep.Classifier} class
   * handles both training and inference.
   */
  //private Classifier classifier
  private Classifier pushPopClassifier;
  private Classifier conceptIDClassifier;
  private Classifier binaryClassifier;
  private Classifier arcConnectClassifier;
  private Classifier pushIndexClassifier;
  private CacheTransition system;

  private final Config config;

  /**
   * Language used to generate
   * {@link edu.stanford.nlp.trees.GrammaticalRelation} instances.
   */
  private final Language language;

  AMRParser() {
    this(new Properties());
  }

  public void loadIndex(Map<String, Integer> indexer, List<String> labels, String file, boolean isLabel) {
    int line_no = 0;
    //indexer = new HashMap<>();

    for (String line: IOUtils.readLines(file)) {
      line = line.trim();
      if (line.equals(""))
        break;
      if (line_no == 0) {
        int n_toks = Integer.parseInt(line);
        if (isLabel) {
          for (int i = 0; i < n_toks; i++)
            labels.add("O");
        }
          //labels = Arrays.asList(new String[n_toks]);
      }
      else {
        String[] splits = line.split(" ");
        if (splits.length != 2) {
          System.err.println("extra space found!");
          System.exit(1);
        }
        int index = Integer.parseInt(splits[1]);
        indexer.put(splits[0], index);
        if (isLabel)
          labels.set(index, splits[0]);
      }
      line_no++;
    }
    //return ret;
  }

  public void initIndexer(Map<String, Integer> featIndexer, Map<String, Integer> labelIndexer,
                          List<String>labels, Map<Integer, Map<Integer, Integer>>emissionScores,
                          String index_dir) {
    String feat_file = index_dir + "/feat_index.txt";
    String label_file = index_dir + "/label_index.txt";
    String weight_file = index_dir + "/emissions.txt";
    int line_no = 0;
    loadIndex(featIndexer, null, feat_file, false);
    loadIndex(labelIndexer, labels, label_file, true);

    //emissionScores = new HashMap<>();

    for (String line: IOUtils.readLines(weight_file)) {
      line = line.trim();
      if (line.equals(""))
        break;
      String[] splits = line.split(" ");
      if (line_no == 0) {
        int numfeats = Integer.parseInt(splits[0]);
        int numLabels = Integer.parseInt(splits[1]);
      }
      else {
        int featIndex = Integer.parseInt(splits[0]);
        int labelIndex = Integer.parseInt(splits[1]);

        if (splits.length < 3) {
          System.err.println(line_no);
          System.err.println(line);
          System.err.println(weight_file);
        }
        if (!emissionScores.containsKey(featIndex)) {
          emissionScores.put(featIndex, new HashMap<>());
        }
        emissionScores.get(featIndex).put(labelIndex, Integer.parseInt(splits[2]));
      }
      line_no++;
    }
  }

  public void loadConceptIDChoices(String conceptIDDict, String lemmaIDDict) {
    word2choices = new HashMap<>();
    lemma2choices = new HashMap<>();
    int nullIndex = conceptIDLabelIndexer.get("conID:-NULL-");
    for (String line: IOUtils.readLines(conceptIDDict)) {
      line = line.trim();
      if (!line.equals("")) {
        String[] splits = line.split(" #### ");
        String word = splits[0].trim();
        if (word2choices.containsKey(word)) {
          System.err.println("Weird, repeated word found: " + word);
          System.exit(1);
        }
        word2choices.put(word, new HashSet<>());

        int threshold = 0;

        if (splits[1].split(" ").length > 12)
          threshold = 20;

        int maxCount = 0;
        for (String s: splits[1].split(" ")) {
          String concept = s.split(":")[0];
          int count = Integer.parseInt(s.split(":")[1]);
          if (count <= threshold)
            continue;

          if (count > maxCount)
            maxCount = count;
          String target = "conID:" + concept;

          if (!conceptIDLabelIndexer.containsKey(target)) {
            System.err.println("conceptID unseen target:" + target);
            continue;
          }
          int l = conceptIDLabelIndexer.get(target);
          word2choices.get(word).add(l);
        }

        if (word2choices.containsKey(word)) {
          if (maxCount < 15) {
            Set<Integer>choices = word2choices.get(word);
            if (choices.size() == 1 && choices.contains(nullIndex))
              word2choices.remove(word);
          }
        }
      }
    }

    for (String line: IOUtils.readLines(lemmaIDDict)) {
      line = line.trim();
      if (!line.equals("")) {
        String[] splits = line.split(" #### ");
        String lemma = splits[0].trim();
        if (lemma2choices.containsKey(lemma)) {
          System.err.println("Weird, repeated lemma found: " + lemma);
          System.exit(1);
        }
        lemma2choices.put(lemma, new HashSet<>());

        int threshold = 0;

        int maxCount = 0;

        if (splits[1].split(" ").length > 12)
          threshold = 20;

        for (String s: splits[1].split(" ")) {
          String concept = s.split(":")[0];
          int count = Integer.parseInt(s.split(":")[1]);
          if (count <= threshold)
            continue;


          String target = "conID:" + concept;

          if (!conceptIDLabelIndexer.containsKey(target)) {
            System.err.println("conceptID unseen target:" + target);
            continue;
          }

          if (count > maxCount)
            maxCount = count;

          int l = conceptIDLabelIndexer.get(target);
          lemma2choices.get(lemma).add(l);
        }
        if (lemma2choices.containsKey(lemma)) {
          if (maxCount < 15) {
            Set<Integer>choices = lemma2choices.get(lemma);
            if (choices.size() == 1 && choices.contains(nullIndex))
              lemma2choices.remove(lemma);
          }
        }

      }
    }
  }

  public boolean isPredicate(String concept) {
    int length = concept.length();
    if (length < 3)
      return false;
    if (!(concept.charAt(concept.length()-3) == '-'))
      return false;
    String digits = "0123456789";
    return digits.contains(concept.substring(length-1));
  }

  public String getConceptCategory(String concept, Map<String, Set<String>> conceptArcChoices) {
    if (conceptArcChoices.containsKey(concept))
      return concept;
    if (concept.equals("NE") || concept.contains("NE_"))
      return "NE";
    if (concept.equals("VB_VERB") || concept.equals("DATE") || concept.equals("NUMBER") || constSet.contains(concept))
      return concept;
    if (concept.contains("VB_NOUN"))
      return "VB_NOUN";
    if (isPredicate(concept))
      return "PRED-01";
    return "OTHER";
  }

  public void loadDefaultArcCandidates() {
    arcLabelDefaultCandidates = new HashSet<>();
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("L-ARG0"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("L-ARG1"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("L-ARG2"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("L-mod"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("R-ARG0"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("R-ARG1"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("R-ARG2"));
    arcLabelDefaultCandidates.add(arcLabelLabelIndexer.get("R-mod"));

  }

  public void loadArcMap(String file, Map<String, Set<String>> conceptArcChoices, double threshold) {
    for (String line: IOUtils.readLines(file)) {
      line = line.trim().replace("  ", " ");
      if (!line.equals("")) {
        String[] splits = line.split(" ");
        if (splits.length != 2) {
          System.err.println("Wrong arc map length:" + line);
          System.exit(1);
        }
        Map<String, Double> relCounts = new HashMap<>();
        double totalCount = 0.0;
        Set<String> choices = new HashSet<>();
        for (String r: splits[1].trim().split(";")) {
          String l = r.split(":")[0];
          int count = Integer.parseInt(r.split(":")[1]);
          totalCount += count;
        }

        double ceiling = totalCount * (1-threshold);
        double currCount = 0.0;
        for (String r: splits[1].trim().split(";")) {
          if (currCount > ceiling)
            break;
          currCount += Integer.parseInt(r.split(":")[1]);
          String l = r.split(":")[0];
          choices.add(l);
        }
        conceptArcChoices.put(splits[0].trim(), choices);
      }
    }
  }

  public void loadIndexer(String index_dir) {


    String conceptid_dir = index_dir + "/conceptid_dir";
    conceptIDFeatIndexer = new HashMap<>();
    conceptIDLabelIndexer = new HashMap<>();
    conceptIDLabels = new ArrayList<>();
    conceptIDEmissionScores = new HashMap<>();
    initIndexer(conceptIDFeatIndexer, conceptIDLabelIndexer, conceptIDLabels, conceptIDEmissionScores, conceptid_dir);

    String conIDUNK = "conID:" + Config.UNKNOWN;
    conceptIDLabelIndexer.put(conIDUNK, conceptIDLabels.size());
    conceptIDLabels.add(conIDUNK);

    String pushpop_dir = index_dir + "/pushpop_dir";
    pushPopFeatIndexer = new HashMap<>();
    pushPopLabelIndexer = new HashMap<>();
    pushPopLabels = new ArrayList<>();
    pushPopEmissionScores = new HashMap<>();
    initIndexer(pushPopFeatIndexer, pushPopLabelIndexer, pushPopLabels, pushPopEmissionScores, pushpop_dir);

    String pushindex_dir = index_dir + "/pushindex_dir";
    pushIndexFeatIndexer = new HashMap<>();
    pushIndexLabelIndexer = new HashMap<>();
    pushIndexLabels = new ArrayList<>();
    pushIndexEmissionScores = new HashMap<>();
    initIndexer(pushIndexFeatIndexer, pushIndexLabelIndexer, pushIndexLabels, pushIndexEmissionScores, pushindex_dir);

    String arcbinary_dir = index_dir + "/arcbinary_dir";
    arcBinaryFeatIndexer = new HashMap<>();
    arcBinaryLabelIndexer = new HashMap<>();
    arcBinaryLabels = new ArrayList<>();
    arcBinaryEmissionScores = new HashMap<>();
    initIndexer(arcBinaryFeatIndexer, arcBinaryLabelIndexer, arcBinaryLabels, arcBinaryEmissionScores, arcbinary_dir);

    String arclabel_dir = index_dir + "/arclabel_dir";
    arcLabelFeatIndexer = new HashMap<>();
    arcLabelLabelIndexer = new HashMap<>();
    arcLabelLabels = new ArrayList<>();
    arcLabelEmissionScores = new HashMap<>();
    initIndexer(arcLabelFeatIndexer, arcLabelLabelIndexer, arcLabelLabels, arcLabelEmissionScores, arclabel_dir);

    //Set<String> defaultLabelSet =
  }

  public AMRParser(Properties properties) {
    connectWordDistToCount = new HashMap<>();
    nonConnectDistToCount = new HashMap<>();
    depConnectDistToCount = new HashMap<>();
    depNonConnectDistToCount = new HashMap<>();

    config = new Config(properties);
    wordConceptCounts = new HashMap<>();
    lemmaConceptCounts = new HashMap<>();
    mleConceptID = new HashMap<>();
    lemMLEConceptID = new HashMap<>();
    conceptIDDict = new HashMap<>();
    unalignedSet = new ArrayList<>();

    this.language = config.language;

  }

  /**
   * Get an integer ID for the given word. This ID can be used to index
   * into the embeddings.
   *
   * @return An ID for the given word, or an ID referring to a generic
   *         "unknown" word if the word is unknown
   */
  public int getWordID(String s) {
    return wordIDs.containsKey(s) ? wordIDs.get(s) : wordIDs.get(Config.UNKNOWN);
  }

  public int getLemmaID(String s) {
    return lemIDs.containsKey(s) ? lemIDs.get(s) : lemIDs.get(Config.UNKNOWN);
  }

  public int getConceptID(String s) {
    return conceptIDs.containsKey(s) ? conceptIDs.get(s) : conceptIDs.get(Config.UNKNOWN);
  }

  public int getPosID(String s) {
      return posIDs.containsKey(s) ? posIDs.get(s) : posIDs.get(Config.UNKNOWN);
  }

  public int getDepID(String s) {
    return depIDs.containsKey(s) ? depIDs.get(s) : depIDs.get(Config.UNKNOWN);
  }

  public int getArcID(String s) {
    return arcIDs.containsKey(s) ? arcIDs.get(s) : arcIDs.get(Config.UNKNOWN);
  }

  //We extract the concept ID features for each example
  public String conceptIDFeats(AMRConfiguration c, Set<String>tokSet, Set<String> lemSet) {
    String ret = "";

    //Get word, lemma and POS features for the current word
    int currWordPos = -1;
    if (c.buffer.size() > 0)
      currWordPos= c.buffer.get(0);
    for (int i = currWordPos - 2; i <= currWordPos + 2; i++) {
      int relative = i - currWordPos;
      if (i < 0 || (i >= c.wordSeq.length) || (c.buffer.size() == 0)) {
        ret += (" wrd" + relative + "=" + Config.NULL);
        ret += (" lem" + relative + "=" + Config.NULL);
        ret += (" pos" + relative + "=" + Config.NULL);
      }
      else {
        ret += (" wrd" + relative + "=" + c.wordSeq[i]);
        ret += (" lem" + relative + "=" + c.lemmaSeq[i]);
        ret += (" pos" + relative + "=" + c.posSeq[i]);
      }
    }

    for (int i = 0; i < 2; i++) {
      int currConceptIndex = c.graph.concepts.size()-1-i;
      if (currConceptIndex < 0) {
        ret += (" lc" + i + "=" + Config.NULL);
        ret += (" lcat" + i+ "=" + Config.NULL);
      }
      else {
        String currConcept = c.graph.concepts.get(currConceptIndex).value();
        ret += (" lc" + i+ "=" + currConcept);
        String currCategory = getConceptCategory(currConcept);
        ret += (" lcat" + i+ "=" + currCategory);
      }
    }

    if (currWordPos >= 0) {
      String currWord = c.wordSeq[currWordPos];
      String currLem = c.lemmaSeq[currWordPos];
      if (tokSet.contains(currWord)) {
        ret += " retok=1";
      }
      else
        ret += " retok=0";

      if (lemSet.contains(currLem)) {
        ret += " relem=1";
      }
      else
        ret += " relem=0";
      tokSet.add(currWord);
      lemSet.add(currLem);
    }
    else {
      ret += " retok=0";
      ret += " relem=0";
    }

    return ret;
  }

  public String pushPopFeats(AMRConfiguration c) {
    String ret = "";
    Pair<Integer, Integer> p = c.cache.get(c.cacheSize-1);
    int wordIndex = p.first;
    int cacheIndex = p.second;
    String word = c.getWord(wordIndex);
    ret += (" rmw=" + word);
    String concept = c.getConcept(cacheIndex);
    ret += (" rmc=" + concept);

    for (int i = 1; i <= 2; i++) {
      int relative = -i;
      Pair<Integer, Integer> curr_p = c.cache.get(c.cacheSize-1-i);
      int currWordIndex = curr_p.first;
      int currConceptIndex = curr_p.second;
      word = c.getWord(currWordIndex);
      concept = c.getConcept(currConceptIndex);
      ret += (" rmw" + relative+ "="+ word);
      ret += (" rmc" + relative+ "=" + concept);

      String currArc = c.getArcLabel(currWordIndex, wordIndex);
      ret += (" cdep" + relative+ "=" + currArc);
    }

    if (c.buffer.size() == 0) {
      ret += (" bw=" + Config.NULL);
      ret += (" bl=" + Config.NULL);
      ret += (" bp=" + Config.NULL);
    }
    else {
      int bufferPos = c.buffer.get(0);
      ret += (" bw=" + c.wordSeq[bufferPos]);
      ret += (" bl=" + c.lemmaSeq[bufferPos]);
      ret += (" bp=" + c.posSeq[bufferPos]);
    }

    int numConnect = c.getRightMostConnect();
    //Here we set a threshold of 4 dependency connections
    if (numConnect >= 4)
      numConnect = 4;

    for (String s: c.getRightMostArcs())
      ret += (" dep=" + s);

    ret += (" nCon=" + numConnect);

    return ret;
  }

  public String arcBinaryFeats(AMRConfiguration c, int cacheIndex) {
    String ret = "";
    Pair<Integer, Integer> conceptP = c.lastP;

    int wordIndex = conceptP.first;
    int conceptIndex = conceptP.second;

    //Candidate word
    String word = c.getWord(wordIndex);
    ret += (" cw=" + word);
    String lem = c.getLemma(wordIndex);
    ret += (" cl=" + lem);
    String pos = c.getPOS(wordIndex);
    ret += (" cp=" + pos);
    String concept = c.getConcept(conceptIndex);
    ret += (" cc=" + concept);

    //Cache word
    Pair<Integer, Integer> cacheP = c.cache.get(cacheIndex);
    int cacheWordIndex = cacheP.first;
    int cacheConceptIndex = cacheP.second;
    word = c.getWord(cacheWordIndex);
    ret += (" caw=" + word);
    lem = c.getLemma(cacheWordIndex);
    ret += (" cal=" + lem);
    pos = c.getPOS(cacheWordIndex);
    ret += (" cap=" + pos);
    concept = c.getConcept(cacheConceptIndex);
    ret += (" cac=" + concept);

    //Get all the connected ARCs for candidate word
    for (String l: c.graph.getAllArcs(conceptIndex)) {
      ret += (" carc=" + l);
    }

    //Get all the connected ARCs for the cache word
    for (String l: c.graph.getAllArcs(cacheConceptIndex)) {
      ret += (" caarc=" + l);
    }

    ret += (" arc=" + c.getArcLabel(cacheWordIndex, wordIndex));

    int wordDist = wordIndex - cacheWordIndex;
    if (wordDist <= 0) {
      System.err.println("Weird word distance");
      System.exit(1);
    }
    if (wordDist > 5)
      wordDist = 5;
    ret += (" wdist=" + wordDist);
    int depDist = c.tree.getDepDist(wordIndex, cacheWordIndex);
    ret += (" ddist=" + depDist);

    return ret;
  }

  public String getConceptCategory(String concept) {
    Set<String> digits = new HashSet();
    digits.add("0");
    digits.add("1");
    digits.add("2");
    digits.add("3");
    digits.add("4");
    digits.add("5");
    digits.add("6");
    digits.add("7");
    digits.add("8");
    digits.add("9");

    if (concept.equals("NUMBER"))
      return concept;
    if (concept.contains("VB_VERB"))
      return "VB_VERB";
    if (concept.contains("VB_"))
      return "VB";
    if (concept.equals("DATE"))
      return "DATE";
    if (concept.contains("NE_") || concept.equals("NE"))
      return "NE";

    if (concept.equals("and") || concept.equals("or"))
      return "CONJ";

    if (concept.length() < 3)
      return "OTHER";
    String lastDigit = concept.substring(concept.length()-1);
    String lastSecond = concept.substring(concept.length()-2, concept.length()-1);
    String lastThird = concept.substring(concept.length()-3, concept.length()-2);
    if (lastThird.equals("-") && digits.contains(lastDigit) && digits.contains(lastSecond))
      return "VERB";

    return "OTHER";
  }

  public String arcConnectFeats(AMRConfiguration c, int cacheIndex) {
    String ret = "";

    Pair<Integer, Integer> conceptP = c.lastP;

    int wordIndex = conceptP.first;
    int conceptIndex = conceptP.second;

    String firstword, secondword;

    //Candidate word
    String word = c.getWord(wordIndex);
    ret += (" cw=" + word);
    firstword = word;
    String lem = c.getLemma(wordIndex);
    ret += (" cl=" + lem);
    String pos = c.getPOS(wordIndex);
    ret += (" cp=" + pos);
    String concept = c.getConcept(conceptIndex);
    ret += (" cc=" + concept);
    String currCategory = getConceptCategory(concept);
    ret += (" ccat=" + currCategory);
    for (String l: c.getAllChildren(wordIndex))
      ret += (" cdep=" + l);

    //Cache word
    Pair<Integer, Integer> cacheP = c.cache.get(cacheIndex);
    int cacheWordIndex = cacheP.first;
    int cacheConceptIndex = cacheP.second;
    word = c.getWord(cacheWordIndex);
    secondword = word;
    ret += (" caw=" + word);
    lem = c.getLemma(cacheWordIndex);
    ret += (" cal=" + lem);
    pos = c.getPOS(cacheWordIndex);
    ret += (" cap=" + pos);
    concept = c.getConcept(cacheConceptIndex);
    ret += (" cac=" + concept);
    currCategory = getConceptCategory(concept);
    ret += (" cacat=" + currCategory);
    for (String l: c.getAllChildren(cacheWordIndex))
      ret += (" cadep=" + l);

    ret += (" arc=" + c.getArcLabel(cacheWordIndex, wordIndex));

    //Get all the connected ARCs for candidate word
    for (String l: c.graph.getAllArcs(conceptIndex)) {
      ret += (" carc=" + l);
    }

    //Get all the connected ARCs for the cache word
    for (String l: c.graph.getAllArcs(cacheConceptIndex)) {
      ret += (" caarc=" + l);
    }

    int wordDist = wordIndex - cacheWordIndex;
    if (wordDist <= 0) {
      System.err.println("Weird word distance");
      System.exit(1);
    }
    if (wordDist > 4)
      wordDist = 4;
    ret += (" wdist=" + wordDist);
    int depDist = c.tree.getDepDist(wordIndex, cacheWordIndex);
    ret += (" ddist=" + depDist);


    return ret;
  }

  public String pushIndexFeats(AMRConfiguration c) {
    String ret = "";
    for (int i = 0; i < c.cacheSize; i++) {
      int cacheWordIndex = c.cache.get(i).first;
      int cacheConceptIndex = c.cache.get(i).second;
      String word = c.getWord(cacheWordIndex);
      ret += (" ca"+ i +"w=" + word);
      String lem = c.getLemma(cacheWordIndex);
      ret += (" ca"+ i +"l=" + lem);
      String pos = c.getPOS(cacheWordIndex);
      ret += (" ca"+ i +"p=" + pos);
      String concept = c.getConcept(cacheConceptIndex);
      ret += (" ca"+ i +"c=" + concept);
    }
    return ret;
  }

  //Define a separate set of features for pushing index to an index in the cache.
  public List<Integer> pushIndexFeatures(AMRConfiguration c) {
    List<Integer> feature = new ArrayList<>();
    c.getCacheFeats(c.cacheSize, conceptIDs, feature, false);
    c.getCacheFeats(c.cacheSize, wordIDs, feature, true);
    //c.getBufferFeats(c.cacheSize, posIDs, feature, false);
    return feature;
  }

  public Set<Integer> pushIndexBinary(AMRConfiguration c) {
    Set<Integer> feature = new HashSet<>();

    Pair<Integer, Integer> conceptP = c.lastP;

    int wordIndex = conceptP.first;
    int conceptIndex = conceptP.second;

    int depSize = depIDs.size();
    //First dependency children of generated word
    int start_index = depIDs.get(Config.UNKNOWN);
    Set<String> depChildren = c.getAllChildren(wordIndex);
    for (String l: depChildren)
      feature.add(getDepID(l)-start_index);

    //Then all the arcs that has been generated for the current node
    List<String> arcChildren = c.graph.getAllArcs(conceptIndex);
    start_index = arcIDs.get(Config.UNKNOWN);
    if (arcChildren != null)
    for (String l: arcChildren)
      feature.add(getArcID(l)-start_index+depSize);

    return feature;
  }

  public int getType(String oracle) {
    int ret;
    if (oracle.equals("POP") || oracle.contains("conID") || oracle.contains("conGen") || oracle.contains("conEMP")) {
      ret = 0;
    }
    else if (oracle.contains("ARC"))
      ret = 1;
    else //The push index transitions
      ret = 2;
    return ret;
  }

  public boolean checkGold(List<Integer> labels) {
    int oneNum = 0;
    for (int l: labels) {
      if (l == 1) {
        oneNum++ ;
      }
    }
    return oneNum == 1;
  }

  //Feature extraction for training examples in the oracle sequence
  public AMRData genTrainExamples(List<String[]> sents, List<String[]> lemmas, List<String[]> poss, List<DependencyTree> trees, List<AMRGraph> graphs, String outputDir, boolean isTrain, boolean sample, Set<Integer> trainSelected) {
    Set<Integer> overset = new HashSet<>();
    overset.add(1227);
    overset.add(1841);
    overset.add(3225);
    overset.add(5705);
    overset.add(7332);
    overset.add(8498);
    overset.add(9846);
    overset.add(14589);

    try {

      String pushPopFeatFile;
      String conceptIDFeatFile;
      String arcBinaryFeatFile;
      String arcLabelFeatFile;
      String pushIndexFeatFile;

      if (isTrain) {
        pushPopFeatFile = outputDir + "/trainPushPopFeat.txt";
        conceptIDFeatFile = outputDir + "/trainConceptIDFeat.txt";
        arcBinaryFeatFile = outputDir + "/trainArcBinaryFeat.txt";
        arcLabelFeatFile = outputDir + "/trainArcLabelFeat.txt";
        pushIndexFeatFile = outputDir + "/trainPushIndexFeat.txt";
      }
      else {
        pushPopFeatFile = outputDir + "/devPushPopFeat.txt";
        conceptIDFeatFile = outputDir + "/devConceptIDFeat.txt";
        arcBinaryFeatFile = outputDir + "/devArcBinaryFeat.txt";
        arcLabelFeatFile = outputDir + "/devArcLabelFeat.txt";
        pushIndexFeatFile = outputDir + "/devPushIndexFeat.txt";
      }

      pushPopFeatWriter = IOUtils.getPrintWriter(pushPopFeatFile);
      conceptIDFeatWriter = IOUtils.getPrintWriter(conceptIDFeatFile);
      arcBinaryFeatWriter = IOUtils.getPrintWriter(arcBinaryFeatFile);
      arcLabelFeatWriter = IOUtils.getPrintWriter(arcLabelFeatFile);
      pushIndexFeatWriter = IOUtils.getPrintWriter(pushIndexFeatFile);

      AMRData ret = new AMRData(); //Parameters to be fixed

      System.err.println(Config.SEPARATOR);
      System.err.println("Generate training examples...");

      ret.initExample();

      //trainSelected.clear(); //needs to be commented
      //config.sampleRate = 1.0;
      sample = false;
      config.binaryProb = 1.0;

      //double sampleRate = 0.03; //Here we only use 1/10 of examples each time for training
      Random sampleR = new Random();
      sampleR.setSeed(20170328);

      int usedSents = 0;
      int evalLength = 40;

      for (int i = 0; i < sents.size(); ++i) {
        if (sample) {
          if (trainSelected.contains(i))
            continue;
          //double tmp = sampleR.nextDouble();
          //if (tmp > config.sampleRate)
          //  continue;
        }

        //if (i!= 1)
        //  continue;

        Set<String> tokSet = new HashSet<>();
        Set<String> lemmaSet = new HashSet<>();

        String[] tokSeq = sents.get(i);

        if (!isTrain && tokSeq.length > evalLength) //For dev, we limit sentence length up to 40
          continue;
        else if (isTrain && (tokSeq.length > 60 || overset.contains((i+1)))) //For training, we limit sentence length up to 60
          continue;
        usedSents += 1;

        String[] lemmaSeq = lemmas.get(i);
        String[] posSeq = poss.get(i);
        DependencyTree tree = trees.get(i);
        AMRGraph graph = graphs.get(i);
        graph.setSentence(tokSeq);
        graph.setLemmas(lemmaSeq);

        int tokIndex = 0;

        AMRConfiguration c = new AMRConfiguration(config.CACHE, tokSeq.length);
        c.wordSeq = tokSeq;
        c.lemmaSeq = lemmaSeq;
        c.posSeq = posSeq;
        c.tree = tree;
        //c.tree.buildDists(config.depDistThreshold);
        c.setGold(graph);

        //for (int j = 0; j < graph.concepts.size(); j++) {
        //  ConceptLabel curr = graph.concepts.get(j);
        //  System.err.println(curr.value() + ":" + curr.aligned);
        //}

        //System.exit(1);

        c.startAction = true;

        Random random = new Random();
        long startT = System.currentTimeMillis();

        //Here for each sentence, a sequence of actions to take
        while (!system.isTerminal(c)) {
          String oracle = system.getOracle(c, tokIndex);
          String nextWord = null;

          //Actions processing the buffer.
          if (oracle.contains("conID") || oracle.contains("conEMP")) {
            nextWord = tokSeq[tokIndex];
            tokIndex += 1;
          }

          if (System.currentTimeMillis() - startT > 1000.0) {//Have been in this example for over 1s, jump out
            System.err.println("Sentence " + (i + 1));
            System.err.println("Over time " + StringUtils.join(tokSeq, " "));

            break;
          }

          int oracleType = getType(oracle);

          if (!oracle.contains("ARC")) {

            if (oracle.contains("conGen")) { //****Special here, has excluded
              system.apply(c, oracle);
              continue;
            }

            //First push or pop decision
            List<Integer> feature;
            Set<Integer> binFeatures;
            String featStr = "";
            if (oracle.equals("POP")) {
              //feature = pushPopFeatures(c);
              //binFeatures = pushPopBinary(c);
              String pushPopFeatStr = pushPopFeats(c);
              featStr = "POP " + pushPopFeatStr;
              pushPopFeatWriter.write(featStr+ "\n");

              //ret.addExample(feature, binFeatures, 1, 4, -1);
              system.apply(c, oracle);

              continue;
            }
            if (oracle.contains("PUSH")) {
              String pushIndexFeatStr = pushIndexFeats(c);
              featStr = oracle + pushIndexFeatStr;
              pushIndexFeatWriter.write(featStr+ "\n");
              //feature = pushIndexFeatures(c);
              //binFeatures = pushIndexBinary(c);
              //System.err.println("PUSH Index, feature size: " + feature.size());
            } else {
              //First use an example for the binary choice push
              //List<Integer> ppFeatures = pushPopFeatures(c);
              String pushPopFeatStr = pushPopFeats(c);
              featStr = "PUSH " + pushPopFeatStr;
              //binFeatures = pushPopBinary(c);
              pushPopFeatWriter.write(featStr+ "\n");

              //ret.addExample(ppFeatures, binFeatures, 0, 4, -1); //Separately process push binary example
              String cidFeatStr = conceptIDFeats(c, tokSet, lemmaSet);
              featStr = oracle + " " + cidFeatStr;
              conceptIDFeatWriter.write(featStr + "\n");

              //feature = conceptIDFeatures(c);
              //binFeatures = conceptIDBinary(c);
              //System.err.println("Concept Identification, feature size: " + feature.size());
            }

            int label = system.getTransitionID(oracle, oracleType);

            int wordID = -1;
            if (oracle.contains("conID") || oracle.contains("conEMP")) {
              String currWord = tokSeq[c.getBuffer(0)];
              if (!currWord.equals(nextWord)) {
                System.err.println("Something wrong with buffer!");
                System.exit(1);
              }
              wordID = getWordID(currWord);

              Set<Integer> target = conceptIDDict.get(wordID);
              if (target.size() <= 1) { //If the example has only one possible choice, continue
                system.apply(c, oracle);
                continue;
              }
            }

            if (label == -1) { //Edges to be processed independently
              System.err.println(label);
              System.err.println("Label error!");
              System.err.println(oracle);
              System.err.println(system.transitionList(oracleType));
              System.exit(1);
            }
            system.apply(c, oracle);

            //ret.addExample(feature, binFeatures, label, oracleType, wordID);
          } else { //For predictions of ARCs, need separate procedure
            //First we need to split the oracle into individual edge-connecting decisions
            String[] parts = oracle.split(":");
            String[] arcDecisions = parts[1].split("#");

            if (arcDecisions.length != c.cacheSize) {
              System.err.println("The number of arc decisions does not match cache size!");
              System.exit(1);
            }


            //For each position, we have to divide each position into two separate decisions
            for (int cacheIndex = 0; cacheIndex < c.cacheSize; cacheIndex++) {
              String currAction = arcDecisions[cacheIndex];

              String currTransition = "ARC" + cacheIndex + ":" + currAction;
              if (currAction.equals("O")) {
                double tmp = random.nextDouble();
                if (tmp > config.binaryProb)
                  continue;
              }

              if (c.cache.get(cacheIndex).second == -1)
                continue;

              int cacheWordIndex = c.cache.get(cacheIndex).first;
              int cacheConceptIndex = c.cache.get(cacheIndex).second;
              int currWordIndex = tokIndex - 1;
              if (currWordIndex != c.lastP.first) {
                System.err.println(tokSeq[currWordIndex]);
                System.err.println(tokSeq[c.lastP.first]);
              }

              //if (cacheWordIndex >= 0 && currWordIndex >= 0)
              //System.err.println(tokSeq[currWordIndex] + ":" + tokSeq[cacheWordIndex] + ":" + currAction);

              int wordDist = currWordIndex - cacheWordIndex;

              int depDist = c.tree.getDepDist(currWordIndex, cacheWordIndex);
              if (depDist > 5) {
                depDist = 5;
              }

              //List<Integer> binFeatures = binaryChoiceFeatures(c, cacheIndex);
              //Set<Integer> binaryFeatures = binaryChoiceBinary(c, cacheIndex);

              String arcBinaryFeatStr = arcBinaryFeats(c, cacheIndex);

              int binLabels;
              String featStr = "";
              if (currAction.equals("O")) {
                if (!nonConnectDistToCount.containsKey(wordDist))
                  nonConnectDistToCount.put(wordDist, 0);
                nonConnectDistToCount.put(wordDist, nonConnectDistToCount.get(wordDist) + 1);

                if (!depNonConnectDistToCount.containsKey(depDist))
                  depNonConnectDistToCount.put(depDist, 0);
                depNonConnectDistToCount.put(depDist, depNonConnectDistToCount.get(depDist) + 1);

                featStr = "O" + arcBinaryFeatStr;
                arcBinaryFeatWriter.write(featStr + "\n");

                //binLabels = 0;
                //ret.addExample(binFeatures, binaryFeatures, binLabels, 3, -1);
                continue;
              } else {
                if (!connectWordDistToCount.containsKey(wordDist))
                  connectWordDistToCount.put(wordDist, 0);
                connectWordDistToCount.put(wordDist, connectWordDistToCount.get(wordDist) + 1);

                if (!depConnectDistToCount.containsKey(depDist))
                  depConnectDistToCount.put(depDist, 0);
                depConnectDistToCount.put(depDist, depConnectDistToCount.get(depDist) + 1);

                featStr = "Y" + arcBinaryFeatStr;
                arcBinaryFeatWriter.write(featStr + "\n");
                //binLabels = 1;
                //ret.addExample(binFeatures, binaryFeatures, binLabels, 3, -1);
              }
              //System.err.println("ARC binary choice, feature size: " + binFeatures.size());
              if (currAction.contains(Config.UNKNOWN)) {
                system.apply(c, currTransition);
                continue;
              }

              String arcConnectFeatStr = arcConnectFeats(c, cacheIndex);
              featStr = currAction + arcConnectFeatStr;
              if (currAction.equals("L-ARG0") && featStr.contains("cc=each") && featStr.contains("cac=NE_country")) {
                System.err.println("Sentence " + i);
                c.goldGraph.printGraph();
                System.err.println(StringUtils.join(tokSeq, " "));
                System.err.println(featStr);
                System.exit(1);
              }
              arcLabelFeatWriter.write(featStr + "\n");
              //binaryFeatures = arcConnectBinary(c, cacheIndex);

              system.apply(c, currTransition);

              int label = system.getTransitionID(currAction, 1);

              if (label == -1) { //Edges to be processed independently
                System.err.println("ARC Label error!");
                System.err.println(currAction);
                System.err.println(system.transitionList(oracleType));
                System.exit(1);
              }
            }
          }

          //c.printConfig();
        }
        conceptIDFeatWriter.write("\n");
      }

      conceptIDFeatWriter.close();
      pushPopFeatWriter.close();
      arcBinaryFeatWriter.close();
      arcLabelFeatWriter.close();
      pushIndexFeatWriter.close();

      if (isTrain) {
        System.err.println("Training using " + usedSents + " sentences!");
      } else {
        System.err.println("Evaluate using " + usedSents + " sentences!");
      }

      return ret;
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  public List<String> oracleSequence(String[] tokSeq, AMRGraph graph) {

    //String[] posSeq = poss.get(i);
    //  DependencyTree tree = trees.get(i);
    graph.setSentence(tokSeq);

    int tokIndex = 0;
    List<String> ret = new ArrayList<>();

    AMRConfiguration c = new AMRConfiguration(config.CACHE, tokSeq.length);
    c.wordSeq = tokSeq;
    c.setGold(graph);

    c.startAction = true;

    long startT = System.currentTimeMillis();

    while (!system.isTerminal(c)) {
      String oracle = system.getOracle(c, tokIndex);

      //Actions processing the buffer.
      if (oracle.contains("conID") || oracle.contains("conEMP")) {
        tokIndex += 1;
      }

      if (System.currentTimeMillis() - startT > 1000.0) {//Have been in this example for over 1s, jump out
        System.err.println("Over time: " + StringUtils.join(tokSeq, " "));
        break;
      }

      ret.add(oracle);
      system.apply(c, oracle);
    }
    return ret;
  }

  //Generate a map from word to a set of the possible concepts.
  //Maybe should filter the concept mappings for too infrequent concept choices.
  public void generateConceptMap(List<String[]> sents, List<String[]> lemmas, List<String[]> poss, List<AMRGraph> graphs) {
    int sent_id = 0;
    Counter<String> unalignConceptCounter = new IntCounter<>();

    for (AMRGraph g : graphs) {

      String[] tokSeq = sents.get(sent_id);
      String[] lemmaSeq = lemmas.get(sent_id);
      String[] posSeq = poss.get(sent_id);

      Set<Integer> aligned_set = new HashSet<>();

      for (ConceptLabel c : g.concepts) {
        //int conceptID = -1;
        String concept = c.value();
        if (c.aligned) {
          //String conceptTran = "conID:" + concept;
          //if (!conceptIDTargetIDs.containsKey(conceptTran))
          //  conceptIDTargetIDs.put(conceptTran, conceptIDTargetIDs.size());

          //conceptID = conceptIDTargetIDs.get(conceptTran);
          if (c.alignments.size() != 1) {
            System.err.println("Alignment size is not one");
            System.exit(1);
          }
          for (int index : c.alignments) {
            aligned_set.add(index);
            String currWord = tokSeq[index];
            String currLem = lemmaSeq[index];
            if (wordIDs.containsKey(currWord)) {
              int wordID = wordIDs.get(currWord);
              if (!wordConceptCounts.containsKey(wordID))
                wordConceptCounts.put(wordID, new HashMap<>());
              Map<String, Integer> conceptCounts = wordConceptCounts.get(wordID);
              if (conceptCounts.containsKey(concept))
                conceptCounts.put(concept, conceptCounts.get(concept) + 1);
              else
                conceptCounts.put(concept, 1);
            }

            if (!lemmaConceptCounts.containsKey(currLem))
              lemmaConceptCounts.put(currLem, new HashMap<>());
            Map<String, Integer> conceptCounts = lemmaConceptCounts.get(currLem);
            if (conceptCounts.containsKey(concept))
              conceptCounts.put(concept, conceptCounts.get(concept) + 1);
            else
              conceptCounts.put(concept, 1);
          }
        }
        else {
          unalignConceptCounter.incrementCount(concept);
        }
      }

      for (int i = 0; i < tokSeq.length; i++) {
        if (aligned_set.contains(i))
          continue;
        int wordID = wordIDs.get(tokSeq[i]);
        String lem = lemmaSeq[i];
        if (!wordConceptCounts.containsKey(wordID))
          wordConceptCounts.put(wordID, new HashMap<>());
        if (!lemmaConceptCounts.containsKey(lem))
          lemmaConceptCounts.put(lem, new HashMap<>());
        Map<String, Integer> conceptCounts = wordConceptCounts.get(wordID);
        //int nullID = conceptIDs.get(config.NULL);
        if (conceptCounts.containsKey(config.NULL))
          conceptCounts.put(config.NULL, conceptCounts.get(config.NULL)+1);
        else
          conceptCounts.put(config.NULL, 1);

        conceptCounts = lemmaConceptCounts.get(lem);
        if (conceptCounts.containsKey(config.NULL))
          conceptCounts.put(config.NULL, conceptCounts.get(config.NULL)+1);
        else
          conceptCounts.put(config.NULL, 1);
      }

      sent_id += 1;
    }
    List<String> sortedUnalign = Counters.toSortedList(unalignConceptCounter, false);
    unalignedSet = freqFilter(unalignConceptCounter, sortedUnalign, 1);
    ratioFilter();
  }

  public List<String> freqFilter(Counter<String>counts, List<String> sortedList, int freq) {
    List<String> ret = new ArrayList<>();
    for (String s: sortedList) {
      if (counts.getCount(s) < freq)
        break;
      ret.add(s);
    }
    return ret;
  }

  //Adjust the alignment to avoid possible alignment errors
  public void ratioFilter() {
    int ne_index = getWordID("NE");
    //System.err.println("NE test:" + knownWords.get(ne_index));
    for(int wordID: wordConceptCounts.keySet()) {
      Map<String, Integer> newMap = new HashMap<>();
      Map<String, Integer> oldMap = wordConceptCounts.get(wordID);

      if (wordID == ne_index)
        continue;

      int maxCount = 0;
      String maxConcept = null;
      for (String concept: oldMap.keySet()) {
        int currCount = oldMap.get(concept);
        if (currCount > maxCount) {
          maxCount = currCount;
          maxConcept = concept;
        }
      }
      if (maxConcept == null) {
        System.err.println("Wrong with word: " + knownWords.get(wordID));
        System.err.println(wordConceptCounts.get(wordID));
        System.exit(1);
      }
      mleConceptID.put(wordID, maxConcept); //The mle concept map for a word
      int threshold = (int)(maxCount * config.conceptIDRatio);
      for (String concept: oldMap.keySet()) {
        int currCount = oldMap.get(concept);
        if (currCount >= threshold) {
          newMap.put(concept, currCount);
        }
      }
      wordConceptCounts.put(wordID, newMap);
    }
    //System.err.println(wordConceptCounts.get(ne_index));

    for(String lemma: lemmaConceptCounts.keySet()) {
      Map<String, Integer> newMap = new HashMap<>();
      Map<String, Integer> oldMap = lemmaConceptCounts.get(lemma);

      int maxCount = 0;
      String maxConcept = null;
      for (String concept: oldMap.keySet()) {
        int currCount = oldMap.get(concept);
        if (currCount > maxCount) {
          maxCount = currCount;
          maxConcept = concept;
        }
      }
      if (maxConcept == null) {
        System.err.println("Wrong with lemma: " + lemma);
        System.err.println(lemmaConceptCounts.get(lemma));
        System.exit(1);
      }

      //ratio /= 1.5;
      lemMLEConceptID.put(lemma, maxConcept); //The mle concept map for a word
      int threshold = (int)(maxCount * config.lemmaConceptIDRatio);
      for (String concept: oldMap.keySet()) {
        int currCount = oldMap.get(concept);
        if (currCount >= threshold) {
          newMap.put(concept, currCount);
        }
      }
      lemmaConceptCounts.put(lemma, newMap);
    }
  }
  /**
   * Generate unique integer IDs for all known words / part-of-speech
   * tags / dependency relation labels.
   *
   * All three of the aforementioned types are assigned IDs from a
   * continuous range of integers; all IDs 0 <= ID < n_w are word IDs,
   * all IDs n_w <= ID < n_w + n_pos are POS tag IDs, and so on.
   */
  private void generateIDs() {
    wordIDs = new HashMap<>();
    lemIDs = new HashMap<>();
    posIDs = new HashMap<>();
    depIDs = new HashMap<>();
    conceptIDs= new HashMap<>();
    arcIDs = new HashMap<>();

    int index = 0;
    for (String word : knownWords)
      wordIDs.put(word, (index++));
    for (String lem: knownLemmas)
      lemIDs.put(lem, (index++));
    for (String pos : knownPos)
      posIDs.put(pos, (index++));
    for (String dep: knownDeps)
      depIDs.put(dep, (index++));
    for (String concept : knownConcepts)
      conceptIDs.put(concept, (index++));
    for (String arc : knownArcs)
      arcIDs.put(arc, (index++));
  }

  /**
   * Scan a corpus and store all words, part-of-speech tags, and
   * dependency relation labels observed. Prepare other structures
   * which support word / POS / label lookup at train- / run-time.
   */
  private void genDictionaries(List<String[]> tokens, List<String[]> lemmas, List<String[]> posTags, List<DependencyTree> trees, List<AMRGraph> graphs) {
    // Collect all words (!), etc. in lists, tacking on one sentence
    // after the other
    List<String> word = new ArrayList<>();
    List<String> lem = new ArrayList<>();
    List<String> pos = new ArrayList<>();
    List<String> dep = new ArrayList<>();
    List<String> concepts = new ArrayList<>();
    List<String> arcs = new ArrayList<>();

    for (String[] tokSeq : tokens)
      for (String tok : tokSeq)
        word.add(tok);

    for (String[] lemSeq : lemmas)
      for (String lemma : lemSeq)
        lem.add(lemma);

    for (String[] posSeq : posTags)
      for (String tok : posSeq)
        pos.add(tok);

    for (DependencyTree tree: trees) {
      for (String lab: tree.label) {
        dep.add(lab);
      }
    }

    for (AMRGraph graph : graphs) {
      //int rootIndex = graph.root;
      for (ConceptLabel c : graph.concepts) {
        concepts.add(c.value());
        for (String s : c.rels)
          arcs.add(s);
      }
    }

    // Generate "dictionaries," possibly with frequency cutoff
    knownWords = Util.generateDict(word, config.wordCutOff);
    knownLemmas = Util.generateDict(lem);
    knownPos = Util.generateDict(pos);
    knownDeps = Util.generateDict(dep);
    knownConcepts = Util.generateDict(concepts);
    //config.arcCutOff = 10; //needs to be commented
    knownArcs = Util.topDict(arcs, config.arcCutOff);

    knownWords.add(0, Config.UNKNOWN); //Tokens in the buffer to be processed
    knownWords.add(1, Config.NULL);
    knownLemmas.add(0, Config.UNKNOWN);
    knownLemmas.add(1, Config.NULL);
    knownPos.add(0, Config.UNKNOWN);
    knownPos.add(1, Config.NULL);
    knownDeps.add(0, Config.UNKNOWN);
    knownDeps.add(1, Config.NULL);

    knownArcs.add(0, Config.UNKNOWN);
    knownArcs.add(1, Config.NULL); //In case in the current setting there is not

    knownConcepts.add(0, Config.UNKNOWN);
    knownConcepts.add(1, Config.NULL);

    allLabels = new ArrayList<>(knownWords);
    allLabels.addAll(knownLemmas);
    allLabels.addAll(knownPos);
    allLabels.addAll(knownDeps);
    allLabels.addAll(knownConcepts);
    allLabels.addAll(knownArcs);

    generateIDs();

    System.err.println(Config.SEPARATOR);
    System.err.println("#Word: " + knownWords.size());
    System.err.println("#Lemma: " + knownLemmas.size());
    System.err.println("#POS: " + knownPos.size());
    System.err.println("#Deps: " + knownDeps.size());
    System.err.println("#Concepts: " + knownConcepts.size());
    System.err.println("#Arcs: " + knownArcs.size());
  }

  public void saveConceptID(String outputDir) {
    try{
      String wordConIDFile = outputDir + "/conceptIDCounts.dict";
      outputWriter = IOUtils.getPrintWriter(wordConIDFile);
      for (int wordID : wordConceptCounts.keySet()) { //Write down all the ambiguous cases
        String currWord = knownWords.get(wordID);
        Map<String, Integer> conceptCounts = wordConceptCounts.get(wordID);
        //if (conceptCounts.size() < 2) {
        //  continue;
        //}
        String result = "";
        for (String concept : conceptCounts.keySet()) {
          result += (concept + ":" + conceptCounts.get(concept) + " ");
        }
        outputWriter.write(currWord + " #### "+ result + "\n");
      }
      outputWriter.close();

      String lemConIDFile = outputDir + "/lemConceptIDCounts.dict";
      outputWriter = IOUtils.getPrintWriter(lemConIDFile);
      for (String lemma : lemmaConceptCounts.keySet()) { //Write down all the ambiguous cases
        Map<String, Integer> conceptCounts = lemmaConceptCounts.get(lemma);
        //if (conceptCounts.size() < 2) {
        //  continue;
        //}
        String result = "";
        for (String concept : conceptCounts.keySet()) {
          result += (concept + ":" + conceptCounts.get(concept) + " ");
        }
        outputWriter.write(lemma + " #### "+ result + "\n");
      }
      outputWriter.close();
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  public void saveMLEConceptID(String outputDir) {
    try{
      String mleConIDFile = outputDir + "/conceptID.dict";
      outputWriter = IOUtils.getPrintWriter(mleConIDFile);
      for (int wordID : mleConceptID.keySet()) {
        String currWord = knownWords.get(wordID);
        String concept = mleConceptID.get(wordID);
        outputWriter.write(currWord + " #### "+ concept + "\n");
      }
      outputWriter.close();

      String lemMLEConIDFile = outputDir + "/lemConceptID.dict";
      outputWriter = IOUtils.getPrintWriter(lemMLEConIDFile);
      for (String lemma : lemMLEConceptID.keySet()) {
        //String currWord = knownWords.get(wordID);
        String concept = lemMLEConceptID.get(lemma);
        outputWriter.write(lemma + " #### "+ concept + "\n");
      }
      outputWriter.close();
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  public void writeConfusionMatrix(Counter<Integer>appearred, Counter<Integer>retrieved,
                                   Counter<Integer>corretP, List<String> transitions) {

    try {
      //First should sort all transitions according to their frequency
      outputWriter.write("transitionID  transitionStr  numAppearances  numPredictted  numCorrect  precision  recall" + "\n");

      List<Integer> keys = Counters.toSortedList(appearred, false); //Sort according to descending order
      int size = keys.size();
      if (size > 2000) {
        size = 2000;
      }

      for (int i = 0; i < size; i++) {
        int l = keys.get(i);
        //for (int l: appearred.keySet()) {
        String currTransition = transitions.get(l);
        double numAppeared = 0.0;
        if (appearred.containsKey(l)) {
          numAppeared = appearred.getCount(l);
        }
        double numRetrieved = 0;
        if (retrieved.containsKey(l)) {
          numRetrieved = retrieved.getCount(l);
        }
        double numCorrect = 0;
        if (corretP.containsKey(l)) {
          numCorrect = corretP.getCount(l);
        }

        double precision = 0.0;
        double recall = 0.0;
        if (numAppeared != 0) {
          precision = numCorrect / numRetrieved;
        }
        if (numRetrieved != 0) {
          recall = numCorrect / numAppeared;
        }
        outputWriter.write(l + " " + currTransition+ " " + numAppeared + " " + numRetrieved +
                " " + numCorrect+ " "+ precision + " " + recall + "\n");
      }
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  public void writeConfig(AMRConfiguration c) {
    String[] wordSeq = c.wordSeq;
    try {

      for (int i = 0; i < wordSeq.length; i++) {
        //wordID
        String s = Integer.toString(i);

        //word str
        s += (" "+ wordSeq[i]);
        if (c.wordToConcept.containsKey(i)) {
          int conceptID = c.wordToConcept.get(i);
          //conceptID
          s += (" " + conceptID);

          ConceptLabel clabel = c.graph.concepts.get(conceptID);
          //concept str
          s += (" " + clabel.value());

          //rels
          List<String> rs = new ArrayList<>();
          for (int j = 0; j < clabel.rels.size(); j++) {
            String r = clabel.rels.get(j) + ":" + clabel.tails.get(j);
            rs.add(r);
          }
          if (rs.size() > 0) {
            s += (" " + StringUtils.join(rs, "#"));
          } else
            s += " NONE";

          //parRels
          List<String> prs = new ArrayList<>();
          for (int j = 0; j < clabel.parRels.size(); j++) {
            String r = clabel.parRels.get(j) + ":" + clabel.parConcepts.get(j);
            prs.add(r);
          }
          if (prs.size() > 0) {
            s += (" " + StringUtils.join(prs, "#"));
          } else
            s += " NONE";
        }
        else
          s += " NONE NONE NONE NONE";
        outputWriter.write(s + "\n");
      }
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
    //return s;
  }

  public void writeModelFile(String modelFile, Classifier classifier) {
    try {
      double[][] W1 = classifier.getW1();
      double[] b1 = classifier.getb1();
      double[][] W2 = classifier.getW2();
      double[][] E = classifier.getE();

      Writer output = IOUtils.getPrintWriter(modelFile);

      output.write("dict=" + knownWords.size() + "\n");
      output.write("pos=" + knownPos.size() + "\n");
      //output.write("label=" + knownLabels.size() + "\n");
      output.write("embeddingSize=" + E[0].length + "\n");
      output.write("hiddenSize=" + b1.length + "\n");
      output.write("numTokens=" + (W1[0].length / E[0].length) + "\n");
      //output.write("preComputed=" + preComputed.size() + "\n");

      int index = 0;

      // First write word / POS / label embeddings
      for (String word : knownWords) {
        output.write(word);
        for (int k = 0; k < E[index].length; ++k)
          output.write(" " + E[index][k]);
        output.write("\n");
        index = index + 1;
      }
      for (String pos : knownPos) {
        output.write(pos);
        for (int k = 0; k < E[index].length; ++k)
          output.write(" " + E[index][k]);
        output.write("\n");
        index = index + 1;
      }
      for (String label : knownConcepts) {
        output.write(label);
        for (int k = 0; k < E[index].length; ++k)
          output.write(" " + E[index][k]);
        output.write("\n");
        index = index + 1;
      }

      // Now write classifier weights
      for (int j = 0; j < W1[0].length; ++j)
        for (int i = 0; i < W1.length; ++i) {
          output.write("" + W1[i][j]);
          if (i == W1.length - 1)
            output.write("\n");
          else
            output.write(" ");
        }
      for (int i = 0; i < b1.length; ++i) {
        output.write("" + b1[i]);
        if (i == b1.length - 1)
          output.write("\n");
        else
          output.write(" ");
      }
      for (int j = 0; j < W2[0].length; ++j)
        for (int i = 0; i < W2.length; ++i) {
          output.write("" + W2[i][j]);
          if (i == W2.length - 1)
            output.write("\n");
          else
            output.write(" ");
        }

      // Finish with pre-computation info
      //for (int i = 0; i < preComputed.size(); ++i) {
      //  output.write("" + preComputed.get(i));
      //  if ((i + 1) % 100 == 0 || i == preComputed.size() - 1)
      //    output.write("\n");
      //  else
      //    output.write(" ");
      //}

      output.close();
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  // TODO this should be a function which returns the embeddings array + embedID
  // otherwise the class needlessly carries around the extra baggage of `embeddings`
  // (never again used) for the entire training process
  private double[][] readEmbedFile(String embedFile, Map<String, Integer> embedID) {

    double[][] embeddings = null;
    if (embedFile != null) {
      BufferedReader input = null;
      try {
        input = IOUtils.readerFromString(embedFile);
        List<String> lines = new ArrayList<String>();
        for (String s; (s = input.readLine()) != null; ) {
          lines.add(s);
        }

        int nWords = lines.size();
        String[] splits = lines.get(0).split("\\s+");

        int dim = splits.length - 1;
        embeddings = new double[nWords][dim];
        System.err.println("Embedding File " + embedFile + ": #Words = " + nWords + ", dim = " + dim);

        if (dim != config.embeddingSize)
            throw new IllegalArgumentException("The dimension of embedding file does not match config.embeddingSize");

        for (int i = 0; i < lines.size(); ++i) {
          splits = lines.get(i).split("\\s+");
          embedID.put(splits[0], i);
          for (int j = 0; j < dim; ++j)
            embeddings[i][j] = Double.parseDouble(splits[j + 1]);
        }
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      } finally {
        IOUtils.closeIgnoringExceptions(input);
      }
    }

    if (embeddings != null)
      embeddings = Util.scaling(embeddings, 0, 1.0);
    else
      System.out.println("No embeddings loaded!");
    return embeddings;
  }

  /**
   * Train a new dependency parser model.
   *
   * @param trainDir Training data directory
   * @param devDir Development data directory (used for regular UAS evaluation
   *                of model)
   * @param modelFile String to which model should be saved
   * @param embedFile File containing word embeddings for words used in
   *                  training corpus
   */
  public void train(String trainDir, String devDir, String evalDir, String outputDir, String modelFile, String embedFile, boolean feat) {

    System.err.println("Train directory: " + trainDir);
    System.err.println("Dev directory: " + devDir);
    System.err.println("Eval directory: " + evalDir);
    System.err.println("Model File: " + modelFile);
    System.err.println("Embedding File: " + embedFile);
    //System.err.println("Pre-trained Model File: " + preModel);

    Random sample = new Random();
    sample.setSeed(20170328);
    config.depDistThreshold = 10;

    if(feat) {
      List<String[]> trainSents = new ArrayList<>();
      List<String[]> trainLemmas = new ArrayList<>();
      List<String[]> trainPoss = new ArrayList<>();
      List<AMRGraph> trainGraphs = new ArrayList<AMRGraph>();
      List<DependencyTree> trainTrees = new ArrayList<DependencyTree>();
      Util.loadAMRFile(trainDir, trainSents, trainLemmas, trainPoss, trainTrees, trainGraphs);

      for (DependencyTree tree : trainTrees) {
        tree.buildDists(config.depDistThreshold);
        //tree.printDep(trainSents.get(sent_index));
      }

      List<String[]> devSents = new ArrayList<>();
      List<String[]> devLemmas = new ArrayList<>();
      List<String[]> devPoss = new ArrayList<>();
      List<AMRGraph> devGraphs = new ArrayList<AMRGraph>();
      List<DependencyTree> devTrees = new ArrayList<DependencyTree>();

      int nTrain = trainSents.size();
      Set<Integer> devTrain = new HashSet<>();

      genDictionaries(trainSents, trainLemmas, trainPoss, trainTrees, trainGraphs);
      generateConceptMap(trainSents, trainLemmas, trainPoss, trainGraphs);

      saveConceptID(outputDir);
      saveMLEConceptID(outputDir);

      //NOTE: remove -NULL-, and the pass it to ParsingSystem
      List<String> cDict = new ArrayList<>(knownConcepts);
      List<String> lDict = new ArrayList<>(knownArcs);
      Set<String> uSet = new HashSet<>(unalignedSet);

      system = new CacheTransition(cDict, lDict, uSet, config.CACHE);
      conceptIDDict = system.makeTransitions(wordConceptCounts);

      int unkWord = getWordID(Config.UNKNOWN);
      if (!conceptIDDict.containsKey(unkWord)) {
        Set<Integer> unkSet = new HashSet<>();
        int unkCon = system.conIDUNKNOW();
        unkSet.add(unkCon);
        conceptIDDict.put(unkWord, unkSet);
      }

      buildConstSet();

      long time1 = System.currentTimeMillis();

      setupClassifierForTraining(trainSents, trainLemmas, trainPoss, trainTrees, trainGraphs,
              devSents, devLemmas, devPoss, devTrees, devGraphs, embedFile, outputDir, devTrain);  //If has dev, also evaluate on the dev set

      System.err.println("Time for loading examples: " + (System.currentTimeMillis() - time1) / 1000.0 + " (s)");
      for (int i = 0; i < 100; i++) {
        if (!nonConnectDistToCount.containsKey(i))
          nonConnectDistToCount.put(i, 0);
        if (!depNonConnectDistToCount.containsKey(i))
          depNonConnectDistToCount.put(i, 0);
      }

      try {
        String outputFile = "connectDist.txt";
        outputWriter = IOUtils.getPrintWriter(outputFile);
        outputWriter.write("Connect distribution: (word distance)\n");
        for (int i : connectWordDistToCount.keySet()) {
          if (i >= 40)
            continue;
          System.err.println("Word distance " + i + ", connect times: " + connectWordDistToCount.get(i) + ", non connect" +
                  "times: " + nonConnectDistToCount.get(i) + "\n");
        }

        outputWriter.write("\n\nConnect distribution: (dependency distance)\n");
        for (int i : depConnectDistToCount.keySet()) {
          if (i >= 40)
            continue;
          System.err.println("Dependency distance " + i + ", connect times: " + depConnectDistToCount.get(i) + ", non connect" +
                  "times: " + depNonConnectDistToCount.get(i) + "\n");
        }

        outputWriter.close();
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      }
    }
    else {
      List<String[]> evalSents = null;
      List<String[]> evalLemmas = null;
      List<String[]> evalPoss = null;
      List<DependencyTree> evalTrees = null;

      outputActionChoices = new HashSet<>();

      String arcLabelFile = config.indexDir + "/arclabel_output_choices.txt";

      for (String line: IOUtils.readLines(arcLabelFile)) {
        line = line.trim();
        if (!line.equals("")) {
          outputActionChoices.add(line);
        }
      }

      Config.pushIndexTokens = 2 * config.CACHE;

      system = new CacheTransition(null, null, null, config.CACHE);
      buildConstSet();

      loadIndexer(config.indexDir);

      concept2outgo = new HashMap<>();
      concept2income = new HashMap<>();
      String outgo_file = config.indexDir + "/outgoing_edges.txt";
      String income_file = config.indexDir + "/incoming_edges.txt";

      loadArcMap(outgo_file, concept2outgo, 0.01);
      loadArcMap(income_file, concept2income, 0.01);

      loadDefaultArcCandidates();

      String conceptIDDict = config.indexDir + "/conceptIDCounts.dict";
      String lemmaIDDict = config.indexDir + "/lemConceptIDCounts.dict";
      loadConceptIDChoices(conceptIDDict, lemmaIDDict);

      String evalTokFile = evalDir + "/tok";
      String evalLemmaFile = evalDir + "/lem";
      String evalPosFile = evalDir + "/pos";
      String evalDepFile = evalDir + "/dep";
      evalSents = IOUtils.readToks(evalTokFile);
      evalLemmas = IOUtils.readToks(evalLemmaFile);
      evalPoss = IOUtils.readToks(evalPosFile);
      evalTrees = new ArrayList<>();
      Util.loadConllFile(evalDepFile, evalTrees, evalSents);
      for (DependencyTree tree : evalTrees)
        tree.buildDists(config.depDistThreshold);

      //loadIndexer(config.indexDir);
      String outputFile = evalDir + "/eval" + ".txt";
      //String outputFile = outputDir + "/eval" + ".txt";
      String arcStatsFile = evalDir + "/arcStats" + ".txt";
      try {
        outputWriter = IOUtils.getPrintWriter(outputFile);
        Writer arcStats = IOUtils.getPrintWriter(arcStatsFile);
        for (int j = 0; j < evalSents.size(); j++) {
          //if (j > 10)
          //  break;
          String[] tokSeq = evalSents.get(j);
          AMRConfiguration predicted = evaluate(tokSeq, evalLemmas.get(j), evalPoss.get(j), evalTrees.get(j));

          outputWriter.write("Sentence " + j + " : " + StringUtils.join(tokSeq, " ") + "\n");

          //List<String> oracleSequence = oracleSequence(tokSeq, devGraphs.get(j));
          //outputWriter.write("Oracle action sequence: " + StringUtils.join(oracleSequence, " ") + "\n");
          outputWriter.write("Predicted action sequence: " + StringUtils.join(predicted.actionSeq, " ") + "\n");
          writeConfig(predicted);
          outputWriter.write("\n");
          arcStats.write(predicted.numArcs + "\n");
          //predicted.graph.printGraph();
        }
        arcStats.close();
        outputWriter.close();
      } catch (IOException e) {
        throw new RuntimeIOException(e);
      }
    }
  }

  //public void setupClassifierForTraining(List<String[]> trainSents, List<String[]> trainPoss, List<DependencyTree> trainTrees,
  //                                        List<AMRGraph> trainGraphs, String embedFile) {
  //  setupClassifierForTraining(trainSents, trainPoss, trainTrees, trainGraphs, null, null, null, null, embedFile);
  //}

  /**
   * Prepare a classifier for training with the given dataset.
   * Three different classifiers for three different procedures.
   */
  private void setupClassifierForTraining(List<String[]> trainSents, List<String[]>trainLemmas, List<String[]> trainPoss, List<DependencyTree> trainTrees,
                                          List<AMRGraph> trainGraphs, List<String[]> devSents, List<String[]> devLemmas, List<String[]> devPoss,
                                          List<DependencyTree> devTrees, List<AMRGraph> devGraphs, String embedFile, String outputDir, Set<Integer> trainSelected) {

    int numEmbeddings = knownWords.size()+ knownLemmas.size() + knownPos.size()+ knownDeps.size()+
            knownArcs.size()+ knownConcepts.size(); //Number of embeddings
    int pushPopHiddenSize = config.hiddenSize;
    int conceptIDHiddenSize = config.hiddenSize;
    int arcConnectHiddenSize = config.hiddenSize;
    int pushIndexHiddenSize = config.hiddenSize;

    int conceptIDOutputSize = system.numTransitions(0);
    int arcConnectOutputSize = system.numTransitions(1);
    int pushIndexOutputSize = system.numTransitions(2);

    if(knownDeps.size() != depIDs.size()) {
      System.err.println("no way, size different!");
      System.err.println(knownDeps.size());
      System.err.println(depIDs.size());
      System.exit(1);
    }

    if(knownArcs.size() != arcIDs.size()) {
      System.err.println("no way, size different!");
      System.err.println(knownDeps.size());
      System.err.println(depIDs.size());
      System.exit(1);
    }

    int numBinaryFeatures = (knownDeps.size() + knownArcs.size()) * 2 + 20; //Encode the deps and arcs binary features, different for different classfiers

    NeuralParam pushPopParam = new NeuralParam(numEmbeddings, config.embeddingSize, Config.pushPopTokens,
            numBinaryFeatures, pushPopHiddenSize, 2, Config.pushPopTokens);
    pushPopParam.randomInitialize(config.initRange);

    NeuralParam conceptIDParam = new NeuralParam(numEmbeddings, config.embeddingSize, Config.conIDTokens,
            numBinaryFeatures, conceptIDHiddenSize, conceptIDOutputSize, Config.conIDTokens);
    conceptIDParam.randomInitialize(config.initRange);

    NeuralParam binaryParam = new NeuralParam(numEmbeddings, config.embeddingSize, Config.binTokens,
            numBinaryFeatures, arcConnectHiddenSize, 2, Config.binTokens);
    binaryParam.randomInitialize(config.initRange);

    NeuralParam arcConnectParam = new NeuralParam(numEmbeddings, config.embeddingSize, Config.arcConnectTokens,
            numBinaryFeatures, arcConnectHiddenSize, arcConnectOutputSize, Config.arcConnectTokens);
    arcConnectParam.randomInitialize(config.initRange);

    NeuralParam pushIndexParam = new NeuralParam(numEmbeddings, config.embeddingSize, Config.pushIndexTokens,
            numBinaryFeatures, pushIndexHiddenSize, pushIndexOutputSize, Config.pushIndexTokens);
    pushIndexParam.randomInitialize(config.initRange);

    // Read embeddings into `embedID`, `embeddings`
    Map<String, Integer> embedID = new HashMap<String, Integer>();
    double[][] embeddings = readEmbedFile(embedFile, embedID);

    // Try to match loaded embeddings with words in dictionary
    Random random = Util.getRandom();
    int foundEmbed = 0;
    for (int i = 0; i < numEmbeddings; ++i) {
      int index = -1;
      if (i < knownWords.size()) {
        String str = knownWords.get(i);
        if (embedID.containsKey(str)) index = embedID.get(str);
        else if (embedID.containsKey(str.toLowerCase())) index = embedID.get(str.toLowerCase());
      }
      if (index >= 0) {
        ++foundEmbed;
        for (int j = 0; j < config.embeddingSize; ++j) {
          double val = embeddings[index][j];
          conceptIDParam.setEmbedding(i, j, val);
          binaryParam.setEmbedding(i, j, val);
          arcConnectParam.setEmbedding(i, j, val);
          pushIndexParam.setEmbedding(i, j, val);
        }
      } else {
        for (int j = 0; j < config.embeddingSize; ++j) {
          //E[i][j] = random.nextDouble() * config.initRange * 2 - config.initRange;
          //E[i][j] = random.nextDouble() * 0.2 - 0.1;
          //E[i][j] = random.nextGaussian() * Math.sqrt(0.1);
          double val = random.nextDouble() * 0.02 - 0.01;
          conceptIDParam.setEmbedding(i, j, val);
          binaryParam.setEmbedding(i, j, val);
          arcConnectParam.setEmbedding(i, j, val);
          pushIndexParam.setEmbedding(i, j, val);
        }
      }
    }
    System.err.println("Found embeddings: " + foundEmbed + " / " + knownWords.size());

    AMRData trainSet = genTrainExamples(trainSents, trainLemmas, trainPoss, trainTrees, trainGraphs, outputDir, true, true, trainSelected);

    AMRData devSet = null;
    ////if (devSents != null) {
    devSet = genTrainExamples(devSents, devLemmas, devPoss, devTrees, devGraphs, outputDir, false, false, null);
    ////}

    //pushPopClassifier = new Classifier(config, trainSet.pushPopExamples, pushPopParam, null, devSet.pushPopExamples);
    //conceptIDClassifier = new Classifier(config, trainSet.conceptIDExamples, conceptIDParam, conceptIDDict, devSet.conceptIDExamples);
    //arcConnectClassifier = new Classifier(config, trainSet.arcConnectExamples, arcConnectParam, null, devSet.arcConnectExamples);
    //pushIndexClassifier = new Classifier(config, trainSet.pushIndexExamples, pushIndexParam, null, devSet.pushIndexExamples);
    //binaryClassifier = new Classifier(config, trainSet.binaryChoiceExamples, binaryParam, null, devSet.binaryChoiceExamples);

    //trainSet = null;
    //devSet = null;
    System.gc();
  }

  private int argmax(double[] scores, Set<Integer> choices) {
    int opt = -1;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < scores.length; i++) {
      if ((choices!=null) && (!choices.contains(i)))
        continue;
      if (scores[i] > max) {
        opt = i;
        max = scores[i];
      }
    }
    return opt;
  }

  public void buildConstSet() {
    constSet = new HashSet<>();
    constSet.add("NUMBER");
    constSet.add("-");
    constSet.add("interrogative");
    constSet.add("imperative");
    constSet.add("expressive");
  }

  public Set<Integer> getFiltered(AMRConfiguration c, int first, int second) {
    Set<Integer> filtered = new HashSet<>();
    ConceptLabel firstC = c.graph.concepts.get(first);
    ConceptLabel secondC = c.graph.concepts.get(second);
    for (String l: firstC.rels) {
      filtered.add(system.arcTransitionIDs.get("L-"+l));
    }
    for (String l: secondC.rels) {
      filtered.add(system.arcTransitionIDs.get("R-"+l));
    }
    return filtered;
  }

  public void sanity_test() {

  }

  public int computeEmissionScore(Map<Integer, Map<Integer, Integer>>emissionScores, int featIndex, int labelIndex) {
    if (!emissionScores.containsKey(featIndex))
      return 0;
    if (!emissionScores.get(featIndex).containsKey(labelIndex))
      return 0;
    return emissionScores.get(featIndex).get(labelIndex);
  }

  public int argmaxChoice(String[] feats, Map<String, Integer> featIndexer, Set<Integer> candidates,
                          Map<Integer, Map<Integer, Integer>> emissionScores) {
    boolean neverfire = true;
    int choice = -1;
    int maxScore = Integer.MIN_VALUE;
    for (int curr_y: candidates) {
      int currScore = 0;
      for (String s : feats) {
        int featIndex = -1;
        if (featIndexer.containsKey(s)) {
          featIndex = featIndexer.get(s);
        }
        if (featIndex == -1)
          continue;
        currScore += computeEmissionScore(emissionScores, featIndex, curr_y);
        if (currScore != 0)
          neverfire = false;
      }
      if (currScore > maxScore) {
        maxScore = currScore;
        choice = curr_y;
      }
    }
    if (neverfire) {
      System.err.println("features never fired: " + StringUtils.join(feats, " "));

    }
    return choice;
  }

  public Set<Integer> getConceptIDCandidates(String word, String lemma) {
    if (word2choices.containsKey(word))
      return word2choices.get(word);
    if (lemma2choices.containsKey(word))
      return lemma2choices.get(lemma);
    return null;
  }

  private AMRConfiguration evaluate(String[] tokSeq, String[] lemSeq, String[] posSeq, DependencyTree tree) {

    AMRConfiguration c = new AMRConfiguration(config.CACHE, tokSeq.length);

    c.wordSeq = tokSeq;
    c.lemmaSeq = lemSeq;
    c.posSeq = posSeq;
    c.tree = tree;

    Map<Integer, Integer> wordToConcept = new HashMap<>();
    int POP = pushPopLabelIndexer.get("POP");
    int PUSH = pushPopLabelIndexer.get("PUSH");

    Set<Integer> pushPopCandidates = new HashSet<>();
    pushPopCandidates.add(PUSH);
    pushPopCandidates.add(POP);

    int ARC = arcBinaryLabelIndexer.get("Y");
    int NONARC = arcBinaryLabelIndexer.get("O");

    Set<Integer> arcBinaryCandidates = new HashSet<>();
    arcBinaryCandidates.add(ARC);
    arcBinaryCandidates.add(NONARC);

    Set<Integer> pushIndexCandidates = new HashSet<>();
    for (String l: pushIndexLabelIndexer.keySet()) {
      pushIndexCandidates.add(pushIndexLabelIndexer.get(l));
    }

    c.startAction = true;

    Set<String> tokSet = new HashSet<>();
    Set<String> lemmaSet = new HashSet<>();

    while (!system.isTerminal(c)) {
      //List<Integer> feature = null;
      //Set<Integer> binaryFeature = null;
      //int numTrans = -1;
      int opt = -1;
      String optTrans = null;

      if (c.startAction) {
        if (c.buffer.size() == 0) {  //Here we can only pop
          optTrans = "POP";
          c.actionSeq.add(optTrans);
          system.apply(c, optTrans);
          continue;
        }

        if (c.stack.size() != 0) { //If stack size 0, can never pop
          String pushPopFeatStr = pushPopFeats(c).trim();
          String[] pushPopFeats = pushPopFeatStr.split(" ");

          opt = argmaxChoice(pushPopFeats, pushPopFeatIndexer, pushPopCandidates, pushPopEmissionScores);
          //opt = pushPopChoice(pushPopFeats);

          //opt = pushPopClassifier.argmax(feature, binaryFeature, null, config.dropProb);
          if (opt == POP) {
            optTrans = "POP";
            system.apply(c, optTrans);
            c.actionSeq.add(optTrans);
            continue;
          }
        }

        int wordIndex = c.getBuffer(0);
        String currWord = tokSeq[wordIndex];
        String currLemma = lemSeq[wordIndex];
        //String currPos = posSeq[wordIndex];

        Set<Integer> cands = getConceptIDCandidates(currWord, currLemma);
        //System.err.println(cands);
        //System.exit(1);
        String conIDFeatStr = conceptIDFeats(c, tokSet, lemmaSet).trim();
        String[] conIDFeats = conIDFeatStr.split(" ");

        if (cands == null) {
          optTrans = "conID:" + Config.UNKNOWN;
          wordToConcept.put(wordIndex, c.graph.nextConceptId());
        }
        else if (cands.size() == 0) {
          System.err.println("Weird candidates for:" + currWord);
          optTrans = "conID:" + Config.UNKNOWN;
          wordToConcept.put(wordIndex, c.graph.nextConceptId());
        }
        else {
          //opt = conceptIDClassifier.argmax(feature, binaryFeature, choices, config.dropProb);
          if (cands.size() == 1)
            opt = cands.iterator().next();
          else
            opt = argmaxChoice(conIDFeats, conceptIDFeatIndexer, cands, conceptIDEmissionScores);
          if (opt == -1)
            System.err.println(cands);
          optTrans = conceptIDLabels.get(opt);
          //optTrans = system.getTransition(0, opt);
          if (!optTrans.contains(Config.NULL)) //In case the next word is unaligned
            wordToConcept.put(wordIndex, c.graph.nextConceptId());
        }
        system.apply(c, optTrans);
      }
      else if(c.lastAction.equals("conID")) { //ARC connect procedure, decisions of cache size needed
        String[] arcDecisions = new String[config.CACHE];
        //Set<Integer> predictedArcs = new HashSet<>();
        int currWordIndex = c.lastP.first;
        int currConceptIndex = c.lastP.second;
        String conceptLabel = c.getConcept(currConceptIndex);

        String currTransition;

        for (int i = 0; i < config.CACHE; i++) {
          if (c.cache.get(i).second == -1) {
            arcDecisions[i] = "O";
            currTransition = "ARC" + i + ":" + arcDecisions[i];
            system.apply(c, currTransition);
            //System.err.println(currTransition);
            continue;
          }

          int cacheWordIndex = c.cache.get(i).first;
          int cacheConceptIndex = c.cache.get(i).second;
          String cacheConcept = c.getConcept(cacheConceptIndex);

          int depDist = c.tree.getDepDist(currWordIndex, cacheWordIndex);
          int wordDist = currWordIndex - cacheWordIndex;

          if (depDist > 2 || wordDist > 4) {
            arcDecisions[i] = "O";
            currTransition = "ARC" + i + ":" + arcDecisions[i];
            system.apply(c, currTransition);
            continue;
          }

          String arcBinaryFeatStr = arcBinaryFeats(c, i).trim();
          String[] arcBinaryFeats = arcBinaryFeatStr.split(" ");

          opt = argmaxChoice(arcBinaryFeats, arcBinaryFeatIndexer, arcBinaryCandidates, arcBinaryEmissionScores);
          //opt = binaryClassifier.argmax(feature, binaryFeature, null, config.dropProb);
          if (opt == NONARC) {
            arcDecisions[i] = "O";
            currTransition = "ARC" + i + ":" + arcDecisions[i];
            system.apply(c, currTransition);
            continue;
          }

          String conceptOutgoCategory = getConceptCategory(conceptLabel, concept2outgo);
          String conceptIncomeCategory = getConceptCategory(conceptLabel, concept2income);
          String cacheConceptOutgoCategory = getConceptCategory(cacheConcept, concept2outgo);
          String cacheConceptIncomeCategory = getConceptCategory(cacheConcept, concept2income);

          Set<Integer> arcLabelCandidates = new HashSet<>();
          if (concept2outgo.containsKey(conceptOutgoCategory) && concept2income.containsKey(cacheConceptIncomeCategory)) {
            Set<String> incomes = concept2income.get(cacheConceptIncomeCategory);
            for (String l: concept2outgo.get(conceptOutgoCategory)) {
              if (incomes.contains(l)) {
                String currAction = ("L-" + l);
                if (outputActionChoices.contains(currAction)) {
                  arcLabelCandidates.add(arcLabelLabelIndexer.get(currAction));
                }
              }
            }
          }

          if (concept2outgo.containsKey(cacheConceptOutgoCategory) && concept2income.containsKey(conceptIncomeCategory)) {
            Set<String> incomes = concept2income.get(conceptIncomeCategory);
            for (String l: concept2outgo.get(cacheConceptOutgoCategory)) {
              if (incomes.contains(l)) {
                String currAction = ("R-" + l);
                if (outputActionChoices.contains(currAction)) {
                  arcLabelCandidates.add(arcLabelLabelIndexer.get(currAction));
                }
              }
            }
          }

          if (arcLabelCandidates.size() == 0) {
            System.err.println(conceptLabel + " ; " + cacheConcept);
            System.err.println(conceptOutgoCategory + " ; " +cacheConceptIncomeCategory);
            System.err.println(conceptIncomeCategory + " ; " + cacheConceptOutgoCategory);
            System.err.println(concept2income.get(conceptIncomeCategory));
            System.err.println(concept2income.get(cacheConceptIncomeCategory));
            arcLabelCandidates = arcLabelDefaultCandidates;
          }

          String arcLabelFeatStr = arcConnectFeats(c, i).trim();
          String[] arcLabelFeats = arcLabelFeatStr.split(" ");

          //c.numArcs += 1;
          opt = argmaxChoice(arcLabelFeats, arcLabelFeatIndexer, arcLabelCandidates, arcLabelEmissionScores);

          //opt = arcConnectClassifier.filterArgmax(feature, binaryFeature, choices, filtered, config.dropProb);
          arcDecisions[i] = arcLabelLabels.get(opt);
          //arcDecisions[i] = system.getTransition(1, opt);
          currTransition = "ARC" + i + ":" + arcDecisions[i];
          system.apply(c, currTransition);
          //predictedArcs.add(opt);
        }

        //System.err.println(StringUtils.join(arcDecisions, ";"));
      }
      else if(c.lastAction.contains("ARC")) { //PUSH index action.
        String pushIndexFeatStr = pushIndexFeats(c).trim();
        String[] pushIndexFeats = pushIndexFeatStr.split(" ");

        opt = argmaxChoice(pushIndexFeats, pushIndexFeatIndexer, pushIndexCandidates, pushIndexEmissionScores);
        optTrans = pushIndexLabels.get(opt);
        system.apply(c, optTrans);
      }
      else {
        System.err.println("Wrong action found during inference:" + c.lastAction);
        System.exit(1);
      }

      //if (optTrans != null)
      //  System.err.println(optTrans);
      //c.actionSeq.add(optTrans);
    }
    c.wordToConcept = wordToConcept;
    return c;
  }

  /**
   * Determine the AMR graph of the given sentence.
   * <p>
   * for general parsing purposes.
   */
  private AMRConfiguration predictInner(String[] tokSeq, String[] lemSeq, String[] posSeq, DependencyTree tree) {
    //int numTrans = system.numTransitions(2);

    //AMRConfiguration c = system.initialConfiguration(tokSeq);
    AMRConfiguration c = new AMRConfiguration(config.CACHE, tokSeq.length);
    //buildConstSet();

    if (!constSet.contains("NUMBER")) {
      System.err.println("Wrong const set!");
      System.exit(1);
    }
    c.wordSeq = tokSeq;
    c.lemmaSeq = lemSeq;
    c.posSeq = posSeq;
    c.tree = tree;
    //c.tree.buildDists(config.depDistThreshold);

    Map<Integer, Integer> wordToConcept = new HashMap<>();

    c.startAction = true;
    //double []scores;

    while (!system.isTerminal(c)) {
      List<Integer> feature = null;
      Set<Integer> binaryFeature = null;
      int numTrans = -1;
      int opt = -1;
      String optTrans = null;

      Set<Integer> choices = new HashSet<>();
      Set<Integer> filtered = new HashSet<>();

      if (c.startAction) {
        if (c.buffer.size() == 0) {  //Here we can only pop
          optTrans = "POP";
          c.actionSeq.add(optTrans);
          system.apply(c, optTrans);
          continue;
        }
        //feature = pushPopFeatures(c);
        //binaryFeature = pushPopBinary(c);
        //feature = binaryChoiceFeatures(c, i);
        if (c.stack.size() != 0) { //If stack size 0, can never pop
          opt = pushPopClassifier.argmax(feature, binaryFeature, null, config.dropProb);
          if (opt == 1) {
            optTrans = "POP";
            system.apply(c, optTrans);
            c.actionSeq.add(optTrans);
            continue;
          }
        }

        //feature = conceptIDFeatures(c);
        //binaryFeature = conceptIDBinary(c);

        int wordIndex = c.getBuffer(0);
        String currWord = tokSeq[wordIndex];
        int wordID = getWordID(currWord);

        for (int curr: conceptIDClassifier.getDict().get(wordID)) {
          choices.add(curr);
        }

        opt = conceptIDClassifier.argmax(feature, binaryFeature, choices, config.dropProb);
        //opt = argmax(scores, choices);
        optTrans = system.getTransition(0, opt);
        if (!optTrans.contains(Config.NULL)) //In case the next word is unaligned
          wordToConcept.put(wordIndex, c.graph.nextConceptId());
      }
      else if(c.lastAction.equals("conID")) { //ARC connect procedure, decisions of cache size needed
        String[] arcDecisions = new String[config.CACHE];
        //Set<Integer> predictedArcs = new HashSet<>();

        for (int i = 0; i < config.CACHE; i++) {
          if (c.cache.get(i).second == -1) {
            arcDecisions[i] = "O";
            continue;
          }

          //feature = binaryChoiceFeatures(c, i);
          //binaryFeature = binaryChoiceBinary(c, i);
          opt = binaryClassifier.argmax(feature, binaryFeature, null, config.dropProb);
          if (opt == 0) {
            arcDecisions[i] = "O";
            continue;
          }

          int conceptIndex = c.lastP.second;
          String conceptLabel = c.getConcept(conceptIndex);

          int cacheIndex = c.cache.get(i).second;
          String cacheConcept = c.getConcept(cacheIndex);

          if (constSet.contains(conceptLabel) && constSet.contains(cacheConcept)) {
            arcDecisions[i] = "O";
            continue;
          }
          else if (constSet.contains(conceptLabel))
            choices = system.constArcInChoices;
          else if (constSet.contains(cacheConcept))
            choices = system.constArcOutChoices;
          else
            choices = null;

          filtered = getFiltered(c, conceptIndex, cacheIndex);

          //feature = arcConnectFeatures(c, i);
          //binaryFeature = arcConnectBinary(c, i);

          c.numArcs += 1;

          opt = arcConnectClassifier.filterArgmax(feature, binaryFeature, choices, filtered, config.dropProb);
          arcDecisions[i] = system.getTransition(1, opt);
          //predictedArcs.add(opt);
        }
        optTrans = "ARC:" + StringUtils.join(arcDecisions, "#");
      }
      else if(c.lastAction.contains("ARC")) { //PUSH index action.
        feature = pushIndexFeatures(c);
        binaryFeature = pushIndexBinary(c);

        opt = pushIndexClassifier.argmax(feature, binaryFeature, null, config.dropProb);
        optTrans = system.getTransition(2, opt);
        //System.err.println(optTrans);
      }
      else {
        System.err.println("Wrong action found during inference:" + c.lastAction);
        System.exit(1);
      }

      c.actionSeq.add(optTrans);
      system.apply(c, optTrans);
    }
    c.wordToConcept = wordToConcept;
    return c;
  }

  /**
   * Explicitly specifies the number of arguments expected with
   * particular command line options.
   */
  private static final Map<String, Integer> numArgs = new HashMap<>();
  static {
    numArgs.put("textFile", 1);
    numArgs.put("outFile", 1);
  }

  public static void main(String[] args) {
    Properties props = StringUtils.argsToProperties(args, numArgs);
    AMRParser parser = new AMRParser(props);
    System.out.println("Start AMR parser");
    // Train with CoNLL-X data
    if (props.containsKey("trainDir"))
      parser.train(props.getProperty("trainDir"), props.getProperty("devDir"), props.getProperty("evalDir"),
              props.getProperty("outputDir"), props.getProperty("model"), props.getProperty("embedFile"), true);


  }
}
