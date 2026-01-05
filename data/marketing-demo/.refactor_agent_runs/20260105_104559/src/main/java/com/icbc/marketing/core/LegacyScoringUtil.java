package com.icbc.marketing.core;

import java.util.Map;

/**
 * [Legacy Component]
 * A static utility class used across the system for basic scoring.
 * Refactoring Target: High coupling. If this changes, everything breaks.
 */
public class LegacyScoringUtil {

    // Magic number alert: 0.618
    private static final double GOLDEN_RATIO = 0.618;

    /**
     * Calculates a base user score.
     * @deprecated Logic is outdated, should be refactored to handle "Region" context.
     */
    public static double calculateBaseScore(Map<String, Object> flinkFeatures) {
        double txnVolume = (double) flinkFeatures.getOrDefault("last_1h_txn_amt", 0.0);
        int loginCount = (int) flinkFeatures.getOrDefault("login_count_7d", 0);

        // Simple linear hard-coded logic
        return (txnVolume * 0.001) + (loginCount * 10) * GOLDEN_RATIO;
    }

    public static boolean isBlacklisted(String userId) {
        // Mock DB call
        return userId.startsWith("B_");
    }
}