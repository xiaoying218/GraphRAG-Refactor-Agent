package com.icbc.marketing.strategy;

import com.icbc.marketing.core.LegacyScoringUtil;
import java.util.Map;

public abstract class AbstractBaseStrategy implements IPromotionStrategy {

    protected double minScoreThreshold;

    public AbstractBaseStrategy(double minScoreThreshold) {
        this.minScoreThreshold = minScoreThreshold;
    }

    // Common logic shared by all subclasses
    protected boolean passBasicRiskCheck(Map<String, Object> features) {
        // DEPENDENCY ALERT: Calls the legacy static utility
        double currentScore = LegacyScoringUtil.calculateBaseScore(features);
        return currentScore > minScoreThreshold;
    }

    @Override
    public abstract String execute(String userId, Map<String, Object> context);
}