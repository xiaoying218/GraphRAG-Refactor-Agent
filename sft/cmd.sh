python make_sr_sft.py \
  --out_dir ../repos/_sft_out_jpetstore6 \
  --recipe org.openrewrite.staticanalysis.CommonStaticAnalysis \
  --files \
    src/main/java/org/mybatis/jpetstore/domain/Cart.java \
    src/main/java/org/mybatis/jpetstore/domain/CartItem.java \
    src/main/java/org/mybatis/jpetstore/domain/LineItem.java \
    src/test/java/org/mybatis/jpetstore/service/OrderServiceTest.java
