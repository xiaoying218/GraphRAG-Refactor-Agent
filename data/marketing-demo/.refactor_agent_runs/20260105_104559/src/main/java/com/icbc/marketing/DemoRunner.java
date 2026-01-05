package com.icbc.marketing;

import com.icbc.marketing.service.CampaignDecisionEngine;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * [Simulator Entry Point]
 * This class simulates the environment where Flink calls our Engine.
 */
public class DemoRunner {

    public static void main(String[] args) {
        System.out.println(">>> Initializing ICBC Marketing Decision Engine...");
        
        // 1. 模拟加载配置 (假装读取 application.properties)
        loadConfiguration();

        // 2. 初始化引擎 (Service Layer)
        CampaignDecisionEngine engine = new CampaignDecisionEngine();

        // 3. 模拟 Case A: 普通用户 (将被 LegacyScoringUtil 拦截或计算低分)
        System.out.println("\n--- Case A: Processing Regular User ---");
        Map<String, Object> regularUserFeatures = new HashMap<>();
        regularUserFeatures.put("last_1h_txn_amt", 50.0);
        regularUserFeatures.put("login_count_7d", 2);
        regularUserFeatures.put("customer_segment", "REGULAR");
        
        String offerA = engine.decideOffer("U_1001", regularUserFeatures);
        System.out.println("Result for U_1001: " + offerA);

        // 4. 模拟 Case B: 高净值 VIP 用户 (将触发 HighNetWorthStrategy)
        System.out.println("\n--- Case B: Processing VIP User ---");
        Map<String, Object> vipFeatures = new HashMap<>();
        vipFeatures.put("last_1h_txn_amt", 50000.0);
        vipFeatures.put("login_count_7d", 20);
        vipFeatures.put("customer_segment", "VIP"); // 这里的 VIP 字符串对应 Strategy 里的判断
        
        String offerB = engine.decideOffer("U_8888", vipFeatures);
        System.out.println("Result for U_8888: " + offerB);

        // 5. 模拟 Case C: 黑名单用户 (触发 Risk Control)
        System.out.println("\n--- Case C: Processing Blacklisted User ---");
        String offerC = engine.decideOffer("B_BadGuy", regularUserFeatures);
        System.out.println("Result for B_BadGuy: " + offerC);
    }

    private static void loadConfiguration() {
        try (InputStream input = DemoRunner.class.getClassLoader().getResourceAsStream("application.properties")) {
            Properties prop = new Properties();
            if (input == null) {
                System.out.println("Sorry, unable to find application.properties");
                return;
            }
            prop.load(input);
            System.out.println("Loaded Config: Strategy Threshold = " + prop.getProperty("strategy.vip.threshold"));
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}