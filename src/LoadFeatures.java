package es.uam.irg.nlp.am;

import es.uam.irg.utils.InitParams;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;

/**
 * Class to extract the features of each text of a file, separated by \t and store it in a json structure
 */
public class LoadFeatures {

    /**
     * @param filePath path of the file to extract features
     */
	public static void loadFeatures(String filePath) {

       String filePath_out = filePath.substring(0,filePath.length()-4).concat("_features.txt");
    	
       String text;
       String resultText;
       int proposalId;
       int argumentId;
       int clase;
       
       
       System.out.println(">> SIMPLE FEAT-EXTRACTION BEGINS");

       Map<String, Object> params = InitParams.readInitParams();
       String language = (String) params.get("language");
       Map<String, HashSet<String>> linkers = (Map<String, HashSet<String>>) params.get("linkers");
       HashSet<String> validLinkers = linkers.get("validLinkers");
       HashSet<String> invalidLinkers = linkers.get("invalidLinkers");
       System.out.format(">> Analysis language: %s, Valid linkers: %s, Invalid linkers: %s\n",
               language, validLinkers, invalidLinkers);
       
       FeatureExtractor miner = new FeatureExtractor(language, validLinkers, invalidLinkers);
       try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
    	   try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath_out))) {
       	   String line;
           while ((line = br.readLine()) != null) {
               String[] columns = line.split("\t"); 

               if (columns.length >= 1) {
            	   proposalId = Integer.parseInt(columns[0]);
               	   argumentId = Integer.parseInt(columns[1]);
                   text = columns[2]; 
                   clase = Integer.parseInt(columns[3]); 

                   resultText = miner.simpleFeatExtraction(text);
                   resultText = "{\"proposal_id\": "+proposalId+", \"argument_id\": "+ argumentId+resultText.substring(14,resultText.length()-1)+", \"clase\": "+clase+"}";
                   
                   	if (!resultText.isEmpty()) {
   	                	writer.write(resultText);
   	                    writer.newLine();
   	                } else {
   	                    System.err.println(">> The Feature Extractor engine had an unexpected error.");
   	                }
                   
               } else {
                   System.out.println("Invalid line: " + line);
               }
           }
       	} catch (IOException e) {
               System.err.println("Error writing to the file: " + e.getMessage());
           }
       } catch (IOException | NumberFormatException e) {
           e.printStackTrace();
           }
	}
	
	/**
	 * 
	 * @param args command line args
	 */
    public static void main(String[] args) {
    	
    	LoadFeatures.loadFeatures("arguments.txt");
    	LoadFeatures.loadFeatures("arguments_premises_claims.txt");
    	LoadFeatures.loadFeatures("arguments_phrases.txt");
    	LoadFeatures.loadFeatures("arguments_premise_validation.txt");
    	LoadFeatures.loadFeatures("arguments_coherence.txt");
    	LoadFeatures.loadFeatures("arguments_consistence.txt");
    	LoadFeatures.loadFeatures("arguments_persuasion.txt");
    	LoadFeatures.loadFeatures("arguments_emotionalethic.txt");
           
       }
   }
