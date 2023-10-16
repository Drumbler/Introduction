import org.junit.Assert;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.List;


@RunWith(Enclosed.class)
public class TestUtil {

    @RunWith(Parameterized.class)
    public static class SigmoidTest {

        @Parameterized.Parameters(name = "{index}: sigmoid[{0}] = {1} ")
        public static Iterable<Object[]> data() {
            return Arrays.asList(new Object[][]{
                    {0, 0.5},
                    {0.5, 0.6224593312018546},
                    {1, 0.7310585786300049},
                    {1.5, 0.8175744761936437},
                    {2, 0.8807970779778823},
                    {2.5, 0.9241418199787566}});
        }

        private final double input;
        private final double expected;

        public SigmoidTest(double input, double expected) {
            this.expected = expected;
            this.input = input;
        }

        @Test
        public void testSigmoid() {
            double actual = Util.sigmoid(input);
            Assert.assertEquals(expected, actual, 0.0000001);
        }

    }

    public static class DotProductTest {

        @Test
        public void testDotProduct() {
            double actual = Util.dotProduct(new double[]{1, 2, 3}, new double[]{4, 5, 6});
            double expected = 32.0;
            Assert.assertEquals(expected, actual, 0.0000001);
        }

        @Test
        public void testDotProduct2() {
            double actual = Util.dotProduct(new double[]{2, 2, 2}, new double[]{3, 3, 3});
            double expected = 18.0;
            Assert.assertEquals(expected, actual, 0.0000001);
        }

        @Test
        public void testDotProduct3() {
            double actual = Util.dotProduct(new double[]{4, 5, 6}, new double[]{7, 8, 9});
            double expected = 122.0;
            Assert.assertEquals(expected, actual, 0.0000001);
        }


    }

}
