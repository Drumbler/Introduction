import org.junit.Assert;
import org.junit.Test;

public class UtilTest {

    @Test
    public void testSigmoid() {
        double actual = Util.sigmoid(0.1);
        double expected = 0.52497918747894;
        Assert.assertEquals(actual, expected, 0.000001);


    }

    @Test
    public void testDotProduct() {
        double actual = Util.dotProduct(new double[]{1, 2, 3}, new double[]{4, 5, 6});
        double expected = 32.0;
        Assert.assertEquals(actual, expected, 0.0000001);

    }

}
