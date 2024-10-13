import { BusinessExplore } from "@/app/modules/business/BusinessExplore";
import Toolbar from "@/app/modules/Toolbar";
import { Business } from "@/types/BusinessTypes";
import { Metadata } from "next";
export const metadata: Metadata = {
    title: 'For You',
    description: '...',
}

export default function forYouPage() {
    const MOCK_DATA: Business[] = [
        {
            name: "The Greek Taverna",
            description: "Authentic Greek cuisine with a modern twist.",
            review: 4.5,
            price: 3,
            location: "123 Main St, Springfield",
            catagory: "Greek",
            image: "https://example.com/image1.jpg",
            id: "1"
        },
        {
            name: "Pizza Paradise",
            description: "Delicious wood-fired pizzas with fresh ingredients.",
            review: 4.8,
            price: 2,
            location: "456 Elm St, Springfield",
            catagory: "Pizza",
            image: "https://example.com/image3.jpg",
            id: "2"
        },
        {
            name: "Sushi World",
            description: "A wide variety of sushi and sashimi, made to order.",
            review: 4.7,
            price: 4,
            location: "789 Oak St, Springfield",
            catagory: "Sushi",
            image: "https://example.com/image5.jpg",
            id: "3"
        },
        {
            name: "Burger Haven",
            description: "Gourmet burgers with unique toppings and sides.",
            review: 4.3,
            price: 2,
            location: "101 Pine St, Springfield",
            catagory: "Burgers",
            image: "https://example.com/image7.jpg",
            id: "4"
        },
        {
            name: "Taco Fiesta",
            description: "A fiesta of flavors with our authentic Mexican tacos.",
            review: 4.6,
            price: 1,
            location: "202 Maple St, Springfield",
            catagory: "Mexican",
            image: "https://example.com/image9.jpg",
            id: "5"
        }
    ];

    return (
        <div className="bg-[#f4f4f4]">
            <Toolbar />
            <div className="flex justify-center items-center py-20 min-h-screen">
                <BusinessExplore businesses={MOCK_DATA} />
            </div>
        </div>
    );
}