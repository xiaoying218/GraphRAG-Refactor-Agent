package com.icbc.marketing.strategy;

import java.util.Map;

public interface IPromotionStrategy {
    String getStrategyName();

    /**
     * Determines if the strategy applies to the current user context.
     */
    boolean isApplicable(Map<String, Object> realTimeFeatures);

    /**
     * Executes the strategy to generate an offer content.
     */
    String execute(String userId, Map<String, Object> context);
}