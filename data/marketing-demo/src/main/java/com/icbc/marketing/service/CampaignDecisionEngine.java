package com.icbc.marketing.service;

import com.icbc.marketing.core.LegacyScoringUtil;
import com.icbc.marketing.strategy.IPromotionStrategy;
import com.icbc.marketing.strategy.impl.HighNetWorthStrategy;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * [The Core System]
 * Real-time Decision Engine.
 * Responsibilities:
 * 1. Metric Analysis (Consume Flink data)
 * 2. Risk Pre-screening (Safety Guardrails) - 【风控在这里！】
 * 3. Marketing Strategy Execution (Recommendation) - 【推荐在这里！】
 */
public class CampaignDecisionEngine {

    private List<IPromotionStrategy> strategies;

    public CampaignDecisionEngine() {
        this.strategies = new ArrayList<>();
        this.strategies.add(new HighNetWorthStrategy());
    }

    public String decideOffer(String userId, Map<String, Object> realTimeFeatures) {
        System.out.println("Processing decision for user: " + userId);

        // =========================================================
        // Step 1: Risk Control Layer (Pre-computation) 【风控层】
        // =========================================================
        
        // 检查 1: 黑名单风控
        // The dependency graph should catch this reference to LegacyScoringUtil
        if (LegacyScoringUtil.isBlacklisted(userId)) {
            return "BLOCK: User is Blacklisted (Risk Control)";
        }

        // 检查 2: 基础指标门槛风控
        // The prompt will ask to refactor this method signature!
        // [Graph RAG Key Point]: If 'calculateBaseScore' changes, this line breaks.
        double baseScore = LegacyScoringUtil.calculateBaseScore(realTimeFeatures);
        
        if (baseScore < 0) {
            return "BLOCK: Activity Score too low (Metric Filter)";
        }

        // =========================================================
        // Step 2: Marketing Strategy Layer (Recommendation) 【推荐层】
        // =========================================================
        
        for (IPromotionStrategy strategy : strategies) {
            // Strategies ALSO depend on the metrics calculated above
            if (strategy.isApplicable(realTimeFeatures)) {
                return strategy.execute(userId, realTimeFeatures);
            }
        }

        return "DEFAULT: General Savings Promo";
    }
}