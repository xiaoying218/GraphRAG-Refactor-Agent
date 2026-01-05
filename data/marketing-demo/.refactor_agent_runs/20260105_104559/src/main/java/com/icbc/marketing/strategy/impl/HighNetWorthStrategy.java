package com.icbc.marketing.strategy.impl;

import com.icbc.marketing.strategy.AbstractBaseStrategy;
import java.util.Map;

public class HighNetWorthStrategy extends AbstractBaseStrategy {

    public HighNetWorthStrategy() {
        super(800.0); // Set high threshold
    }

    @Override
    public String getStrategyName() {
        return "VIP_WEALTH_MANAGEMENT";
    }

    @Override
    public boolean isApplicable(Map<String, Object> realTimeFeatures) {
        String segment = (String) realTimeFeatures.get("customer_segment");
        // Reuse the logic from parent, which calls LegacyScoringUtil
        return "VIP".equals(segment) && super.passBasicRiskCheck(realTimeFeatures);
    }

    @Override
    public String execute(String userId, Map<String, Object> context) {
        return "Offer: Exclusive Gold Deposit Rate + 2000 points";
    }
}